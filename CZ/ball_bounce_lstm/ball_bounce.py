"""
3D "code" that simulates a ball bouncing around in a box.

Takes inputs, runs the "code" using them, and generates some outputs.

This is not a particularly realistic simulation.

Creates DSV.
"""

import sys
import csv
import argparse



DELIMETER = "%"

# The expected args in the expected order. Expressed here so I only have to type it once-ish.
ARGS = ["output_file", "x_pos_initial", "y_pos_initial", "z_pos_initial",
        "x_vel_initial", "y_vel_initial", "z_vel_initial",
        "gravity", "box_side_length", "group_id", "run_id"]

# In case I need to modify the maestro call
# print("python ball_bounce.py {}".format(" ".join("$({})".format(x.upper()) for x in ARGS)))


def run_ball_bounce(xpos_initial, ypos_initial, zpos_initial,
                    xvel_initial, yvel_initial, zvel_initial,
                    gravity, box_side_length, TICKS_PER_SECOND,
                    RUNTIME, DRAG_COEFF):
    """
    Bounce a ball for RUNTIME seconds, at TICKS_PER_SECOND resolution.

    Distance units are arbitrary, time units are seconds.

    Our shape has a density, reference area, and mass of 1, so acceleration
    due to drag is (dragcoeff*vel^2*1*1)/2=1/a, a=dragcoeff*vel^2/2.

    May or may not be a (very small) spherical cow

    :returns: a timeseries of x,y, and z positions, the final x, y, and z velocities,
              and the number of bounces
    """
    ticks = float(TICKS_PER_SECOND)
    x_pos = []
    y_pos = []
    z_pos = []
    x_pos_instant = float(xpos_initial)
    y_pos_instant = float(ypos_initial)
    z_pos_instant = float(zpos_initial)
    x_vel_instant = xvel_initial/ticks
    y_vel_instant = yvel_initial/ticks
    z_vel_instant = zvel_initial/ticks
    tick_grav = gravity/ticks
    num_bounces = 0
    # In the future, we may return to these being randomizeable
    runtime = RUNTIME
    dragcoeff = DRAG_COEFF

    runtime *= TICKS_PER_SECOND
    tick_drag = abs(dragcoeff/ticks)

    def run_axis_tick(axis_pos, axis_vel, dist):
        """Run a single timestep in a single axis."""
        nonlocal num_bounces
        pos_new = axis_pos + axis_vel
        if (pos_new < 0):
            axis_vel *= -1
            # axis_pos = -pos_new
            axis_pos = 0
            num_bounces += 1
        elif (pos_new > dist):
            axis_vel *= -1
            # axis_pos = dist - (pos_new - dist)
            axis_pos = dist
            num_bounces += 1
        else:
            axis_pos = pos_new
        drag = tick_drag*axis_vel**2/2
        axis_vel = (max(0, axis_vel-drag) if axis_vel > 0 else min(0, axis_vel+drag))
        return (axis_pos, axis_vel)

    time = []

    for step in range(0, runtime):
        x_pos_instant, x_vel_instant = run_axis_tick(x_pos_instant,
                                                     x_vel_instant,
                                                     box_side_length)
        y_pos_instant, y_vel_instant = run_axis_tick(y_pos_instant,
                                                     y_vel_instant,
                                                     box_side_length)
        z_pos_instant, z_vel_instant = run_axis_tick(z_pos_instant,
                                                     z_vel_instant,
                                                     box_side_length)
        x_pos.append(x_pos_instant)
        y_pos.append(y_pos_instant)
        z_pos.append(z_pos_instant)
        y_vel_instant -= gravity
        # Extremely good physics
        if (y_pos_instant == 0 and y_vel_instant < 0):
            y_vel_instant = 0

        time.append(RUNTIME*step/runtime)

    return time, x_pos, y_pos, z_pos, x_pos_instant, y_pos_instant, z_pos_instant, x_vel_instant, y_vel_instant, z_vel_instant, num_bounces


def set_params_and_launch(args):
    """Return a DSV row of randomized inputs and resultant outputs."""
    runtime = RUNTIME  # random.choice([10, 60, 180])
    dragcoeff = DRAG_COEFF  # random.choice([0.01, 0.05, 0.1, 0.25, 1])
    all_args = args+[runtime, dragcoeff]
    for output in bouncing_ball(*all_args):
        all_args.append(output)
    return (all_args)


if __name__ == "__main__":
    """Do a single run of a ball bounce and handle all the I/O associated."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--xpos", "-x", help="initial x position", default=0., type=float)
    parser.add_argument("--ypos", "-y", help="initial y position", default=0., type=float)
    parser.add_argument("--zpos", "-z", help="initial z position", default=0., type=float)
    parser.add_argument("--xvel", "-X", help="initial x velocity", default=0., type=float)
    parser.add_argument("--yvel", "-Y", help="initial y velocity", default=0., type=float)
    parser.add_argument("--zvel", "-Z", help="initial z velocity", default=0., type=float)
    parser.add_argument("--gravity", "-g", help="gravity", type=float, default=9.81)
    parser.add_argument("--box_side_length", "-b", help="length of the box's sides", type=float, default=10)
    parser.add_argument("--runtime", "-r" , help="length of time we let the simualtion run for", type=float, default=20)
    parser.add_argument("--frequency", "--ticks_per_seconds", default=20, help="sampling rate", type=int)
    parser.add_argument("--drag", "-d", type=float, help="drag coefficient", default=0.1)
    parser.add_argument("--output","-o", help="output file")
    parser.add_argument("--group", "-G", help="group id", default="group")
    parser.add_argument("--run", "-R", help="run id", type=str, default=1)

    args = parser.parse_args()
    with open(args.output, 'w') as outfile:
        writer = csv.writer(outfile, delimiter=DELIMETER)
        # The ARGS[1:-1] is because we don't need to write the output file or run_id['s raw number].
        names = ["id"]+ARGS[1:-1]+["time", "x_pos", "y_pos", "z_pos",
                                    "x_pos_final", "y_pos_final", "z_pos_final",
                                    "x_vel_final", "y_vel_final", "z_vel_final",
                                    "num_bounces"]
        writer.writerow(names)
        results = [f"{args.group}_{args.run}"]  # id
        # We're not passing output_dir, group_id, or run_id
        numerical_inputs = [args.xpos, args.ypos, args.zpos,
                            args.xvel, args.yvel, args.zvel,
                            args.gravity, args.box_side_length,
                            ]
        inputs = numerical_inputs + [args.frequency, args.runtime, args.drag]
        results.extend(numerical_inputs)
        results.append(args.group)  # add group_id as an input
        results.extend(run_ball_bounce(*inputs))
        writer.writerow(results)
