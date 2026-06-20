from multiagent_global_planners.multiagent_planner import MultiAgentPlanner
from global_planners.global_planner import MapType
from geometry import Point, compute_direction


class SimpleMulti(MultiAgentPlanner):
    MAP = MapType.GRID

    def compute_path(self):
        agents_abstraction_dict = self.server.collect_agents_info(self.agents)

        agents_solution_path_dict = {}
        for aid, agent_info in agents_abstraction_dict.items():
            start = agent_info.position
            goal = agent_info.destination_location
            dist = start.distance(goal)
            if dist < 0.5:
                agents_solution_path_dict[aid] = [goal]
                continue
            step = 0.5
            num_steps = max(1, int(dist / step))
            path = []
            for i in range(1, num_steps + 1):
                t = i / num_steps
                x = start.x + (goal.x - start.x) * t
                y = start.y + (goal.y - start.y) * t
                path.append(Point(x, y))
            agents_solution_path_dict[aid] = path

        return agents_solution_path_dict
