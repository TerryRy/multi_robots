from math import sqrt, sin, cos


class WaypointTracker:
    """
    Executes a sequence of global-frame waypoints by interpolating
    and outputting velocity commands. Pure algorithm, no AI.
    """

    def __init__(self, agent, dt=1.0 / 60):
        self.agent = agent
        self.dt = dt
        self.waypoint_queue = []
        self.current_step = 0
        self._prev_position = None

    def set_waypoints(self, global_waypoints):
        self.waypoint_queue = list(global_waypoints)
        self.current_step = 0
        self._prev_position = None

    def compute_velocity(self, position):
        if len(self.waypoint_queue) == 0:
            return (0.0, 0.0)

        target = self.waypoint_queue[0]
        dx = target[0] - position.x
        dy = target[1] - position.y
        dist = sqrt(dx * dx + dy * dy)

        if dist < 0.03:
            self.waypoint_queue.pop(0)
            if len(self.waypoint_queue) == 0:
                return (0.0, 0.0)
            target = self.waypoint_queue[0]
            dx = target[0] - position.x
            dy = target[1] - position.y
            dist = sqrt(dx * dx + dy * dy)

        max_step = self.agent.cruise_speed * self.dt * 0.8
        speed = min(self.agent.cruise_speed, dist / self.dt * 0.8)
        speed = min(speed, max_step / self.dt)

        norm = max(0.001, dist)
        vx = dx / norm * speed
        vy = dy / norm * speed
        return (vx, vy)

    def has_waypoints(self):
        return len(self.waypoint_queue) > 0
