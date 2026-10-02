from pmsm_gui.app import SimulationApp


def main() -> None:
    app = SimulationApp()
    app.mainloop()


if __name__ == "__main__":
    main()


# Continuous ODE (what you use — solve_ivp / Radau)
# Pros:

# Adaptive step size — solver takes small steps during fast transients and large steps during slow steady state automatically
# Well suited for stiff systems like PMSM where electrical and mechanical dynamics differ by orders of magnitude
# No need to choose a fixed time step — the solver handles it internally
# Mathematically exact representation of the continuous physics
# Error is controlled by rtol and atol tolerances

# Cons:

# Slower per step due to implicit solving and error estimation overhead
# Less intuitive to implement for someone coming from a controls background
# Harder to synchronise with discrete PWM switching events exactly


# Discrete ODE (fixed time step, e.g. Euler, RK4)
# Pros:

# Simple to implement and easy to understand
# Faster per step since no error estimation is needed
# Naturally matches digital controller implementation — real embedded controllers run at fixed sampling rates
# Easier to synchronise with PWM periods exactly
# Better for real-time or hardware-in-the-loop simulation

# Cons:

# Fixed step size must be small enough for the fastest dynamics — for PMSM this means very small steps, making it slow overall
# No automatic adaptation — if the step is too large the solution diverges
# Stiff systems like PMSM require extremely small steps with explicit methods, making them computationally expensive
# Simple methods like Euler accumulate error over time


# ==============================================================================================================
# Yes. The output arrays are all the same length because they are produced by resampling the ODE 
# solution onto a fixed uniform time grid of 10,000 samples per second using numpy.interp. 
# The ODE solver internally uses adaptive step sizes, but the final output is always interpolated 
# onto this fixed grid, so every time-series array has exactly the same number of points.


# Mechanical equation (eq. 3.14):
#   J * d(omega_m)/dt = T_e - T_L
#
# Assume T_e = K_T * i_q* and neglect load (T_L = 0):
#   J * d(omega_m)/dt = K_T * i_q*
#
# Take Laplace transform (d/dt -> s, zero initial conditions):
#   J * s * omega_m(s) = K_T * i_q*(s)
#
# Rearrange to get transfer function:
#   omega_m(s) / i_q*(s) = K_T / (J * s)  ->  eq. 3.15



# TODO: what are my states???
# The 13 ODE states
# Your state vector x has exactly 13 elements, all integrated continuously by 
# Radau from t=0 to tfinal. Initial conditions are all zero except the two thermal states, 
# which start at T_amb.

# The states split into five logical groups:
# x[0..1] — d/q axis stator currents. These are the fast electrical dynamics — time constant ~L/R, 
#           often under 1 ms. They're what makes the system stiff.
# x[2..3] — mechanical speed and angle. Slow dynamics, governed by J * dω/dt = T_e − T_load.
# x[4..6] — PI integrators for the d-axis, q-axis, and speed controllers. These are pure controller states, 
#           not physical ones. They accumulate error and are governed by the anti-windup law.
# x[7..8] — stator and magnet temperatures. Very slow thermal dynamics. T_s affects R_s; T_m affects phi_f.
# x[9..12] — filtered versions of the scheduled inputs (speed reference, mass, slope, Crr). 
#            These exist because schedule steps are discontinuous but the ODE solver needs smooth inputs. 
#            Each filters toward its target with a first-order time constant (tau_omega_ref = 5 ms, tau_mass/slope/crr = 50–150 ms).




# Imagine you're trying to drag a magnet around in a circle using another magnet in your hand, 
# without touching it. You hold your magnet a little bit ahead of the one on the table, so it gets 
# pulled toward yours — and as it moves, you keep shifting your hand forward too, always staying 
# just a bit ahead. Do this right and the table magnet chases yours smoothly around the circle.

# Now here's the important bit: how fast you move your hand isn't something you get to pick freely. 
# It has to match how fast the table magnet is actually moving, at every moment. If you move your 
# hand too fast, you get too far ahead — your pulling magnet ends up beside it, or behind it, or 
# even pulling it the wrong way, and instead of a smooth pull you get a jerky, useless one. If you 
# move too slow, same problem in reverse. The only way to get a clean, strong pull is to constantly 
# watch where the table magnet is, and keep your hand exactly a bit ahead of that — not going at some 
# speed you decided on your own in advance.

# That's your rotor and your stator field. The rotor is the table magnet. The rotating field the stator 
# produces is your hand-magnet. For that field to actually pull the rotor around usefully (instead of 
# jerking it around uselessly), it has to always stay positioned a bit ahead of wherever the rotor 
# physically is right now — which means whoever's "moving the hand" has to know the rotor's position and 
# adjust accordingly, live, all the time.

# Now — a regular wall outlet is like a machine that swings your hand-magnet around at one unchangeable pace, 
# 60 times a second, forever, no matter what the table magnet is doing. It never looks at the table magnet at 
# all. That's fine only in the lucky case where the table magnet already happens to be moving at exactly that 
# pace. It can't help it start from a standstill, and it can't slow down or speed up to stay "just ahead" if 
# the table magnet's speed changes.

# An inverter is the opposite kind of machine: instead of one fixed pace, it's built to watch where the rotor 
# is (or figure it out) and move its "hand" at whatever pace is needed, moment to moment, to always stay just 
# ahead of it — slow at first, faster as the rotor speeds up, whatever's required. That watching-and-adjusting 
# is exactly what "frequency being an output, not an input" means: the pace comes from tracking the rotor, not 
# from a dial someone set beforehand.

# If you hold your hand only a tiny bit ahead, the pull is gentle. If you hold it further ahead (but not too far, 
# or you lose the connection entirely), the pull is much stronger. So to speed up, you widen that ahead-gap for a 
# while — the stronger pull accelerates the rotor faster than it needs just to keep pace, so it speeds up. Once 
# it reaches the new speed you want, you ease the gap back down to just enough to keep it cruising there (fighting 
# only friction/load, not accelerating anymore).

# Your hand's own pace still always matches wherever the rotor currently is, automatically — you're never 
# choosing that pace directly. What you're actually choosing is the size of the gap. Bigger gap now → more 
# pull now → rotor speeds up → your hand's pace (which just tracks the rotor) rises right along with it.

# In your motor, that "gap size" is basically i_q (torque-producing current) — that's the actual knob your 
# speed-loop PI is turning. Widen it, torque goes up, rotor accelerates, frequency rises as a byproduct. 
# Narrow it back down once you're at target speed, and everything settles at the new pace.