from local_planners.local_planner import LocalPlanner
from geometry import compute_direction


class SimpleLocal(LocalPlanner):
    def compute_plan(self, position, velocity, environment, sensor_observation, global_planner_path):
        try:
            if position.arrive(global_planner_path[0]):
                global_planner_path.pop(0)
            goal_pose = global_planner_path[0]
        except IndexError:
            goal_pose = self.agent.destination_location

        if goal_pose is None:
            return (0.0, 0.0)

        speed = compute_direction(position, goal_pose)
        return (speed[0] * self.agent.cruise_speed, speed[1] * self.agent.cruise_speed)
