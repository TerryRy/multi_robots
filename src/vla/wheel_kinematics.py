"""
Kinematics: convert between (v_forward, omega) and (vx, vy, heading)
for differential-drive style control in a holonomic simulator.

The VLA model outputs 8 steps of (v_forward, omega) per robot.
v_forward = forward speed (from expert linear_velocity magnitude)
omega     = angular velocity (from heading change between steps)
"""
from math import cos, sin, atan2, pi


MAX_SPEED = 3.0
MAX_OMEGA = 6.0


def motion_to_control(vx, vy, prev_vx, prev_vy, dt, prev_heading=None):
    """Convert simulated motion (vx,vy from Box2D) to (v_forward, omega).

    v_forward is exact: sqrt(vx²+vy²)
    omega is from heading change: (current_heading - prev_heading) / dt
    heading = atan2(vy, vx) for holonomic, or wheel_heading if tracking.

    Returns:
        (v_forward, omega) — raw, unclipped
    """
    speed = (vx * vx + vy * vy) ** 0.5
    if speed < 1e-6:
        return 0.0, 0.0

    curr_heading = atan2(vy, vx)
    if prev_heading is not None:
        d_heading = curr_heading - prev_heading
        while d_heading > pi:
            d_heading -= 2 * pi
        while d_heading < -pi:
            d_heading += 2 * pi
        omega = d_heading / max(dt, 1e-6)
    else:
        omega = 0.0
    return speed, omega


def control_to_motion(v_forward, omega, heading, dt):
    """Convert (v_forward, omega) back to linear velocity for Box2D.

    Args:
        v_forward: forward speed
        omega: angular velocity (rad/s)
        heading: current robot heading (rad)
        dt: physics timestep

    Returns:
        (vx, vy, new_heading) for setting agent.linear_velocity and body.angle
    """
    new_heading = heading + omega * dt
    vx = v_forward * cos(heading)
    vy = v_forward * sin(heading)
    return vx, vy, new_heading
