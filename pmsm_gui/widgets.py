from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, List, Optional

import numpy as np

from pmsm_gui.settings import CRR_GUIDE

class ScrollableFrame(ttk.Frame):
    '''
    Purpose:
        A reusable Tkinter frame that contains a vertically scrollable area.

    Behavior:
        - Creates a Canvas widget
        - Places an inner frame inside the canvas
        - Adds a vertical scrollbar
        - Keeps the inner frame width synced with the canvas width
        - Supports mouse-wheel scrolling

    Output:
        Creates a GUI widget object. No return value.
    '''

    def __init__(self, master: tk.Misc):
        '''
        Purpose:
            Initializes the scrollable frame and all scrolling behavior.

        Inputs:
            master: parent Tkinter widget

        Output:
            No return value. Builds the widget structure in-place.
        '''
        super().__init__(master)
        canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = ttk.Frame(canvas)
        self.inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        window_id = canvas.create_window((0, 0), window=self.inner, anchor="nw")

        def on_canvas_configure(event: tk.Event) -> None:
            '''
            Purpose:
                Updates the embedded frame width whenever the canvas size changes.

            Inputs:
                event: Tkinter configure event

            Output:
                No return value. Adjusts widget layout.
            '''
            canvas.itemconfig(window_id, width=event.width)

        canvas.bind("<Configure>", on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        def _on_mousewheel(event: tk.Event) -> None:
            '''
            Purpose:
                Handles mouse-wheel scrolling for the canvas.

            Inputs:
                event: Tkinter mouse-wheel event
                - event.delta is the wheel motion amount from Tkinter.
                - Dividing by 120 normalizes one wheel notch, int(...) makes it an integer step count,
                and -1 * flips the direction so scrolling feels natural in the canvas.
                - If event.delta is empty or zero, it uses 0

            Output:
                No return value. Scrolls the visible region vertically.
            '''
            if not canvas.winfo_exists():
                return
            delta = -1 * int(event.delta / 120) if event.delta else 0
            if delta:
                canvas.yview_scroll(delta, "units")

        def _bind_mousewheel(_event: tk.Event) -> None:
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel(_event: tk.Event) -> None:
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", _bind_mousewheel)
        canvas.bind("<Leave>", _unbind_mousewheel)


class ScheduleEditor(tk.Toplevel):
    '''
    Purpose:
        Popup editor window for modifying time-value schedules such as speed,
        mass, slope, and rolling resistance steps.

    Behavior:
        - Displays rows of time/value pairs
        - Lets the user add or delete rows
        - Validates numeric input
        - Sorts schedule entries by time before saving

    Output:
        Creates a top-level editor window.
        Final saved result is stored in self.result as:
            tuple[List[float], List[float]]
        or None if canceled.
    '''

    def __init__(
        self,
        master: tk.Misc,
        title: str,
        time_key: str,
        value_key: str,
        time_label: str,
        value_label: str,
        time_values: List[float],
        step_values: List[float],
        value_validator: Callable[[float], Optional[str]],
        help_kind: Optional[str] = None,
    ):
        '''
        Purpose:
            Builds the schedule editor window and populates it with existing data.

        Inputs:
            master: parent widget
            title: window title
            time_key: internal schedule time key
            value_key: internal schedule value key
            time_label: label text for the time column
            value_label: label text for the value column
            time_values: current list of step times
            step_values: current list of step values
            value_validator: function used to validate each value
            help_kind: optional help mode ("omega" or "crr")

        Output:
            No return value. Initializes the editor UI.
        '''
        super().__init__(master)
        self.title(title)
        self.resizable(True, True)
        self.time_key = time_key
        self.value_key = value_key
        self.value_validator = value_validator
        self.result: Optional[tuple[List[float], List[float]]] = None
        self.rows: List[tuple[tk.StringVar, tk.StringVar]] = []

        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        header = ttk.Frame(self, padding=10)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text=title, font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Button(header, text="Add row", command=self.add_row).grid(row=0, column=1, padx=(8, 0))

        if help_kind:
            help_box = ttk.LabelFrame(self, text="Help", padding=10)
            help_box.grid(row=1, column=0, sticky="ew", padx=10)
            help_box.columnconfigure(0, weight=1)
            ttk.Label(help_box, text=self.build_help_text(help_kind), justify="left", wraplength=760).grid(row=0, column=0, sticky="w")
        else:
            spacer = ttk.Frame(self)
            spacer.grid(row=1, column=0, sticky="ew")

        body = ScrollableFrame(self)
        body.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        self.body = body.inner
        for i, text in enumerate([time_label, value_label, "Action"]):
            ttk.Label(self.body, text=text, font=("Segoe UI", 10, "bold")).grid(row=0, column=i, padx=4, pady=4, sticky="w")

        for t, v in zip(time_values, step_values):
            self.add_row(t, v)

        footer = ttk.Frame(self, padding=10)
        footer.grid(row=3, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Button(footer, text="Cancel", command=self.destroy).grid(row=0, column=1, padx=4)
        ttk.Button(footer, text="Save", command=self.on_save).grid(row=0, column=2, padx=4)

        self.transient(master)
        self.grab_set()

    def build_help_text(self, help_kind: str) -> str:
        '''
        Purpose:
            Returns explanatory text for the selected help type.

        Inputs:
            help_kind:
                - "omega": speed conversion help
                - "crr": rolling-resistance reference values

        Output:
            Returns a string containing help text to display in the editor.
        '''
        if help_kind == "omega":
            # return (
            #     "Useful conversion for the speed-step table:\n"
            #     "vehicle speed [km/h] = 3.6 × (omega_m / gear_ratio) × wheel_radius\n"
            #     "omega_m [rad/s] = (vehicle speed [km/h] / 3.6) × gear_ratio / wheel_radius\n\n"
            #     "Example: with gear_ratio = 9 and wheel_radius = 0.30 m,\n"
            #     "50 rad/s ≈ 6.0 km/h at the vehicle."
            # )
            return (
                "Speed-step values are entered in RPM.\n"
                "Backend motor equations still use rad/s internally.\n\n"
                "Conversion:\n"
                "rad/s = RPM × 2π / 60\n"
                "RPM = rad/s × 60 / (2π)\n"
            )
        if help_kind == "crr":
            lines = ["Typical rolling-resistance coefficients:"]
            for name, val in CRR_GUIDE:
                lines.append(f"• {name}: ~{val:.3f}")
            return "\n".join(lines)
        return ""

    def add_row(self, t_value: float = 0.0, v_value: float = 0.0) -> None:
        '''
        Purpose:
            Adds a new editable schedule row to the table.

        Inputs:
            t_value: initial time value for the row
            v_value: initial step value for the row

        Output:
            No return value. Adds widgets and stores StringVar references.
        '''
        t_var = tk.StringVar(value=str(t_value))
        v_var = tk.StringVar(value=str(v_value))
        row_index = len(self.rows) + 1
        ttk.Entry(self.body, textvariable=t_var, width=16).grid(row=row_index, column=0, padx=4, pady=3, sticky="ew")
        ttk.Entry(self.body, textvariable=v_var, width=16).grid(row=row_index, column=1, padx=4, pady=3, sticky="ew")  # tells Tkinter grid to stretch the widget east and west.
        ttk.Button(self.body, text="Delete", command=lambda idx=row_index - 1: self.delete_row(idx)).grid(row=row_index, column=2, padx=4, pady=3)
        self.rows.append((t_var, v_var))

    def delete_row(self, index: int) -> None:
        '''
        Purpose:
            Deletes a schedule row and rebuilds the visible row widgets.

        Inputs:
            index: zero-based row index to remove

        Output:
            No return value. Updates internal row storage and GUI layout.
        '''
        if len(self.rows) == 1:
            messagebox.showerror("Schedule error", "At least one row must remain.", parent=self)
            return
        self.rows.pop(index)
        for child in self.body.winfo_children():
            info = child.grid_info()
            if int(info.get("row", 0)) > 0:
                child.destroy()
        for i, (t_var, v_var) in enumerate(self.rows, start=1):
            ttk.Entry(self.body, textvariable=t_var, width=16).grid(row=i, column=0, padx=4, pady=3, sticky="ew")
            ttk.Entry(self.body, textvariable=v_var, width=16).grid(row=i, column=1, padx=4, pady=3, sticky="ew")
            ttk.Button(self.body, text="Delete", command=lambda idx=i - 1: self.delete_row(idx)).grid(row=i, column=2, padx=4, pady=3)

    def on_save(self) -> None:
        '''
        Purpose:
            Validates all rows, sorts them by time, and stores the final result.

        Validation:
            - time must be numeric and >= 0
            - value must be numeric
            - value must pass the custom validator

        Output:
            No direct return value.
            On success:
                self.result becomes (times, values)
                and the editor window closes.
            On failure:
                displays an error dialog.
        '''
        times: List[float] = []
        values: List[float] = []
        errors: List[str] = []
        for idx, (t_var, v_var) in enumerate(self.rows, start=1):
            try:
                t = float(t_var.get())
            except ValueError:
                errors.append(f"Row {idx}: time must be a number.")
                continue
            try:
                v = float(v_var.get())
            except ValueError:
                errors.append(f"Row {idx}: value must be a number.")
                continue
            if t < 0:
                errors.append(f"Row {idx}: time must be >= 0.")
            msg = self.value_validator(v)
            if msg:
                errors.append(f"Row {idx}: {msg}")
            times.append(t)
            values.append(v)
        if not errors:
            order = np.argsort(times)
            times = [times[i] for i in order]
            values = [values[i] for i in order]
            self.result = (times, values)
            self.destroy()
        else:
            messagebox.showerror("Schedule error", "\n".join(errors), parent=self)

# =====================================================================================
# inputs information helper
# =====================================================================================
class InfoTooltip:
    """
    Hover/click tooltip that stays inside the application window when possible.
    Hover shows the bubble.
    Leaving hides it unless it was pinned by click.
    Click toggles pinned state.
    """
    def __init__(self, widget: tk.Widget, text: str, wraplength: int = 520):
        self.widget = widget
        self.text = text
        self.wraplength = wraplength
        self.tip_window: Optional[tk.Toplevel] = None
        self.pinned = False

        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide_if_not_pinned)
        widget.bind("<Button-1>", self.toggle)

    def show(self, _event: Optional[tk.Event] = None) -> None:
        if self.tip_window is not None:
            return

        self.tip_window = tk.Toplevel(self.widget)
        self.tip_window.wm_overrideredirect(True)
        self.tip_window.withdraw()

        frame = ttk.Frame(self.tip_window, padding=8, relief="solid", borderwidth=1)
        frame.grid(row=0, column=0, sticky="nsew")

        label = ttk.Label(
            frame,
            text=self.text,
            justify="left",
            wraplength=self.wraplength,
        )
        label.grid(row=0, column=0, sticky="w")

        self.tip_window.update_idletasks()

        tip_w = self.tip_window.winfo_reqwidth()
        tip_h = self.tip_window.winfo_reqheight()

        root = self.widget.winfo_toplevel()
        root.update_idletasks()

        root_x = root.winfo_rootx()
        root_y = root.winfo_rooty()
        root_w = root.winfo_width()
        root_h = root.winfo_height()

        icon_x = self.widget.winfo_rootx()
        icon_y = self.widget.winfo_rooty()
        icon_h = max(self.widget.winfo_height(), 18)

        margin = 18

        usable_left = root_x + margin
        usable_top = root_y + margin
        usable_right = root_x + root_w - margin
        usable_bottom = root_y + root_h - margin

        # Default below-right.
        x = icon_x + 22
        y = icon_y + icon_h + 8

        # Move left if it would exceed the app window.
        if x + tip_w > usable_right:
            x = usable_right - tip_w

        # Move above if it would exceed the app window bottom.
        if y + tip_h > usable_bottom:
            y = icon_y - tip_h - 8

        # Final clamp.
        x = max(usable_left, min(x, usable_right - tip_w))
        y = max(usable_top, min(y, usable_bottom - tip_h))

        self.tip_window.wm_geometry(f"+{int(x)}+{int(y)}")
        self.tip_window.deiconify()

    def hide(self) -> None:
        if self.tip_window is not None:
            self.tip_window.destroy()
            self.tip_window = None

    def hide_if_not_pinned(self, _event: Optional[tk.Event] = None) -> None:
        if not self.pinned:
            self.hide()

    def toggle(self, _event: Optional[tk.Event] = None) -> None:
        self.pinned = not self.pinned
        if self.pinned:
            self.show()
        else:
            self.hide()