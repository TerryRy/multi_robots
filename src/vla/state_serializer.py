"""
State Serializer: converts simulator world state into VLA-compatible
text prompt + per-agent numerical features.
"""
from math import sqrt, sin, cos, atan2, pi


def _rel_angle(dx, dy, agent_angle):
    angle = atan2(dy, dx)
    rel = angle - agent_angle
    while rel > pi:
        rel -= 2 * pi
    while rel < -pi:
        rel += 2 * pi
    return rel


def _downsample_lidar(ray_list, target_bins=16):
    if not ray_list:
        return [0.0] * target_bins
    n = len(ray_list)
    step = max(1, n // target_bins)
    result = []
    for i in range(target_bins):
        start = i * step
        end = min(start + step, n)
        chunk = ray_list[start:end]
        result.append(min(chunk) if chunk else 0.0)
    return result


class StateSerializer:

    def __init__(self, lidar_target_bins=16, max_neighbors=4, sensor_range=4.0):
        self.lidar_target_bins = lidar_target_bins
        self.max_neighbors = max_neighbors
        self.sensor_range = sensor_range

    def serialize(self, simulator):
        agents_data = []
        for agent in simulator.agents:
            agent_body = simulator.b2_objects.get(agent.id)
            if agent_body is None:
                continue

            position = agent.position
            angle = getattr(agent, 'wheel_heading', agent_body.angle)

            goal = None
            goal_type = "none"
            if agent.destination_location is not None:
                goal = agent.destination_location
                if agent.task is not None:
                    task_type = getattr(agent.task, 'type', None)
                    if task_type is not None:
                        goal_type = str(task_type)
                    else:
                        goal_type = "waypoint"

            lidar_down = _downsample_lidar(agent.ray_length_list, self.lidar_target_bins)

            neighbors = self._extract_neighbors(agent, simulator.agents, simulator.b2_objects)

            nearest_obstacle = self._nearest_obstacle(lidar_down, agent.ray_length_list)

            path = []
            if hasattr(agent, 'sequence_of_poses') and agent.sequence_of_poses:
                for p in agent.sequence_of_poses:
                    path.append((p.x, p.y))
                if len(path) > 5:
                    path = path[:5]

            carrying = getattr(agent, 'carrying_item', None)
            has_item = 1 if (carrying is not None and carrying != 0) else 0

            from agents.agent_state_machine import AgentState
            state_str = str(agent.state) if hasattr(agent, 'state') else "CRUISE"

            agents_data.append({
                "id": agent.id,
                "position": (position.x, position.y) if hasattr(position, 'x') else position,
                "angle_deg": angle * 180.0 / pi,
                "angle_rad": angle,
                "speed": (agent_body.linearVelocity[0]**2 + agent_body.linearVelocity[1]**2)**0.5,
                "state": state_str,
                "carrying_item": has_item,
                "destination": (goal.x, goal.y) if goal is not None and hasattr(goal, 'x') else None,
                "goal_type": goal_type,
                "sequence_of_poses": path,
                "lidar": lidar_down,
                "neighbors": neighbors,
                "nearest_obstacle": nearest_obstacle,
            })

        ports_data = []
        for port_list in [simulator.loading_ports, simulator.unloading_ports]:
            for port in port_list:
                port_type = "loading" if port in simulator.loading_ports else "unloading"
                queue_len = 0
                items_remain = 0
                if hasattr(port, 'queue'):
                    queue_len = getattr(port.queue, 'num_agents', lambda: 0)()
                if hasattr(port, 'items'):
                    items_remain = len(port.items)
                ports_data.append({
                    "id": getattr(port, 'identifier', '?'),
                    "type": port_type,
                    "position": (port.location.x, port.location.y) if hasattr(port.location, 'x') else port.location,
                    "queue_len": queue_len,
                    "items_remain": items_remain,
                })

        obstacle_data = []
        if hasattr(simulator, 'environment') and hasattr(simulator.environment, 'obstacles'):
            for obs in simulator.environment.obstacles.values():
                obstacle_data.append({
                    "position": (obs.location.x, obs.location.y),
                    "width": obs.dimension[0],
                    "height": obs.dimension[1],
                })

        task_count = getattr(simulator, 'task_count', 0)
        sim_time = getattr(simulator, 'time', 0)
        aa_col = getattr(simulator, 'agent_agent_collisions', 0)
        ao_col = getattr(simulator, 'agent_static_collisions', 0)

        text_prompt = self._build_text_prompt(
            agents_data, ports_data, obstacle_data, task_count, sim_time,
            aa_col, ao_col,
        )
        features = self._build_features(agents_data, ports_data, obstacle_data)

        return text_prompt, features, agents_data

    def _extract_neighbors(self, agent, all_agents, b2_objects):
        neighbors = []
        agent_pos = agent.position
        if hasattr(agent_pos, 'x'):
            apx, apy = agent_pos.x, agent_pos.y
        else:
            apx, apy = agent_pos

        for other in all_agents:
            if other.id == agent.id:
                continue
            op = other.position
            if hasattr(op, 'x'):
                ox, oy = op.x, op.y
            else:
                ox, oy = op
            dx = ox - apx
            dy = oy - apy
            dist = sqrt(dx * dx + dy * dy)
            if dist <= self.sensor_range:
                body = b2_objects.get(agent.id)
                agent_angle = body.angle if body else 0
                rel_ang = _rel_angle(dx, dy, agent_angle)
                other_body = b2_objects.get(other.id)
                other_heading = other_body.angle if other_body else 0
                neighbors.append({
                    "id": other.id,
                    "dist": dist,
                    "rel_angle_deg": rel_ang * 180.0 / pi,
                    "rel_heading_diff_deg": (other_heading - agent_angle) * 180.0 / pi,
                })

        neighbors.sort(key=lambda n: n["dist"])
        return neighbors[:self.max_neighbors]

    def _nearest_obstacle(self, lidar_down, ray_list):
        if not ray_list:
            return {"dist": 4.0, "rel_angle_deg": 0}
        min_val = min(ray_list)
        if min_val >= 3.99:
            return {"dist": 4.0, "rel_angle_deg": 0}
        idx = ray_list.index(min_val)
        angle = idx / 511.0 * 180.0 - 90
        return {"dist": min_val, "rel_angle_deg": angle}

    def _build_text_prompt(self, agents, ports, obstacles, task_count, sim_time,
                           aa_col=0, ao_col=0):

        agents_unique = []
        seen = set()
        for a in agents:
            if a["id"] not in seen:
                agents_unique.append(a)
                seen.add(a["id"])

        parts = []
        parts.append(f"Timestamp: {sim_time:.2f}s | Packages: {int(task_count)} | "
                     f"Collisions: AA={aa_col} AO={ao_col} | "
                     f"Wheels: {len(agents)}")

        agent_lines = []
        for a in agents_unique:
            pos = a["position"]
            dest = a["destination"]
            dest_str = f"({dest[0]:.1f},{dest[1]:.1f})" if dest else "none"
            lidar_min = min(a["lidar"]) if a["lidar"] else 4.0
            agent_lines.append(f"  Robot_{a['id']}: pos=({pos[0]:.1f},{pos[1]:.1f}) "
                               f"h={a['angle_deg']:.0f} spd={a['speed']:.2f} "
                               f"{a['state']} carry={a['carrying_item']} "
                               f"dest={dest_str} obs={lidar_min:.1f}m")
        parts.extend(agent_lines)

        if ports:
            port_lines = ["Ports:"]
            for p in ports:
                pos = p["position"]
                port_lines.append(f"  {p['id']}({p['type']}): pos=({pos[0]:.1f},{pos[1]:.1f}) "
                                  f"queue={p['queue_len']} items={p['items_remain']}")
            parts.append("\n".join(port_lines))

        if obstacles:
            obs_strs = []
            for o in obstacles:
                obs_strs.append(f"({o['position'][0]:.1f},{o['position'][1]:.1f},"
                                f"w={o['width']:.1f},h={o['height']:.1f})")
            parts.append(f"Obstacles: [{', '.join(obs_strs)}]")

        return "\n".join(parts)

    def _nearest_port(self, agent_pos, ports, agent_heading):
        best_dist, best_angle, best_type = 4.0, 0.0, 0.0
        px, py = agent_pos[0], agent_pos[1]
        for p in ports:
            ppos = p["position"]
            dx = ppos[0] - px
            dy = ppos[1] - py
            dist = sqrt(dx * dx + dy * dy)
            if dist < best_dist:
                best_dist = dist
                rel_angle = atan2(dy, dx) - agent_heading
                while rel_angle > pi:
                    rel_angle -= 2 * pi
                while rel_angle < -pi:
                    rel_angle += 2 * pi
                best_angle = rel_angle
                best_type = 1.0 if p.get("type") == "loading" else 0.0
        return [
            best_dist,
            cos(best_angle), sin(best_angle),
            best_type,
        ]

    def _build_features(self, agents, ports, obstacles):

        state_onehot_map = {
            'AgentState.IDLE': [1, 0, 0, 0, 0, 0],
            'AgentState.LOADING': [0, 1, 0, 0, 0, 0],
            'AgentState.QUEUING': [0, 0, 1, 0, 0, 0],
            'AgentState.HALT': [0, 0, 0, 1, 0, 0],
            'AgentState.CRUISE': [0, 0, 0, 0, 1, 0],
            'AgentState.PREQUEUE': [0, 0, 0, 0, 0, 1],
        }

        max_wheels = 24
        max_ports = 20
        max_obstacles = 10
        max_neighbors = self.max_neighbors
        lidar_bins = self.lidar_target_bins

        def pad_or_trunc(lst, target_len, default):
            result = list(lst[:target_len])
            while len(result) < target_len:
                result.append(default)
            return result

        encoded = []
        for a in agents[:12]:
            pos = a["position"]
            dest = a["destination"]
            dx_goal = dest[0] - pos[0] if dest else 0.0
            dy_goal = dest[1] - pos[1] if dest else 0.0
            dist_goal = sqrt(dx_goal * dx_goal + dy_goal * dy_goal) if dest else 0.0
            heading_rad = a["angle_rad"]

            goal_heading = 0.0
            if dest and dist_goal > 0.001:
                goal_heading = atan2(dy_goal, dx_goal)

            base = [
                pos[0], pos[1],
                cos(heading_rad), sin(heading_rad),
                a["speed"],
                *state_onehot_map.get(a["state"], [0, 0, 0, 0, 0, 0]),
                float(a["carrying_item"]),
                dx_goal, dy_goal,
                cos(goal_heading), sin(goal_heading) if dest else 0.0,
            ]

            neighbor_features = []
            for n in a["neighbors"][:max_neighbors]:
                n_ang = n["rel_angle_deg"] * pi / 180.0
                n_hd = n["rel_heading_diff_deg"] * pi / 180.0
                neighbor_features.extend([
                    n["dist"],
                    cos(n_ang), sin(n_ang),
                    cos(n_hd), sin(n_hd),
                ])
            while len(neighbor_features) < max_neighbors * 5:
                neighbor_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

            lidar_features = pad_or_trunc(a["lidar"], lidar_bins, 4.0)

            near_obs = a["nearest_obstacle"]
            ob_ang_rad = near_obs["rel_angle_deg"] * pi / 180.0
            obstacle_features = [
                near_obs["dist"],
                cos(ob_ang_rad), sin(ob_ang_rad),
            ]

            port_features = self._nearest_port(pos, ports, heading_rad)

            base_feat = base + neighbor_features + lidar_features + obstacle_features + port_features
            # Two wheels per agent: wheel_id=0 (left), wheel_id=1 (right)
            encoded.append(base_feat + [0.0])
            encoded.append(base_feat + [1.0])

        while len(encoded) < max_wheels:
            encoded.append([0.0] * len(encoded[0]) if encoded else [0.0] * 60)

        port_features = []
        for p in ports[:max_ports]:
            port_features.extend([p["position"][0], p["position"][1],
                                  1.0 if p["type"] == "loading" else 0.0,
                                  float(p["queue_len"]), float(p["items_remain"])])
        while len(port_features) < max_ports * 5:
            port_features.extend([0.0] * 5)

        obstacle_features = []
        for o in obstacles[:max_obstacles]:
            obstacle_features.extend([o["position"][0], o["position"][1],
                                       o["width"], o["height"]])
        while len(obstacle_features) < max_obstacles * 4:
            obstacle_features.extend([0.0] * 4)

        return {
            "num_agents": len(agents) * 2,
            "num_wheels": len(agents) * 2,
            "num_ports": len(ports),
            "num_obstacles": len(obstacles),
            "agents": encoded,
            "ports": port_features,
            "obstacles": obstacle_features,
        }
