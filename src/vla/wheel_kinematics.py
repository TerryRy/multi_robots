"""
Wheel kinematics: convert between (left, right) wheel velocities
and (vx, vy, heading) for differential-drive robots.

wheel_base = 0.3m (from DD_planner / HRVO parameters)
max_wheel_speed ≈ 2.0 m/s per wheel
"""
from math import cos, sin, atan2, pi

WHEEL_BASE = 0.3
MAX_WHEEL_SPEED = 3.0
MAX_OMEGA = 4.0


def wheels_to_velocity(left, right, heading):
    """Convert differential-drive wheel speeds to linear velocity vector.

    Args:
        left, right: wheel velocities (m/s)
        heading: current robot heading (rad)

    Returns:
        (vx, vy, new_heading) — vx,vy are linear velocities,
        new_heading is heading updated by angular velocity * 1 sec
    """
    left = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, left))
    right = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, right))
    v_forward = (left + right) / 2.0
    omega = (right - left) / WHEEL_BASE
    vx = v_forward * cos(heading)
    vy = v_forward * sin(heading)
    return vx, vy, omega


def velocity_to_wheels(vx, vy, prev_vx, prev_vy, dt):
    """Convert linear velocity and heading change to wheel speeds.

    Used during data collection to convert expert planner (holonomic)
    outputs into differential-drive wheel commands.

    Args:
        vx, vy: current linear velocity
        prev_vx, prev_vy: previous step's linear velocity
        dt: time between steps

    Returns:
        (left, right) wheel velocities (m/s)
    """
    speed = (vx * vx + vy * vy) ** 0.5
    if speed < 1e-6:
        return 0.0, 0.0

    curr_heading = atan2(vy, vx)
    prev_heading = atan2(prev_vy, prev_vx)
    d_heading = curr_heading - prev_heading
    while d_heading > pi:
        d_heading -= 2 * pi
    while d_heading < -pi:
        d_heading += 2 * pi

    omega = d_heading / max(dt, 1e-6)
    omega = max(-MAX_OMEGA, min(MAX_OMEGA, omega))
    left = speed - omega * WHEEL_BASE / 2.0
    right = speed + omega * WHEEL_BASE / 2.0
    return left, right
