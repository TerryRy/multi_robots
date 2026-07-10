"""
Expert replay simulation: verifies collected data by directly feeding
expert waypoints (stored in JSONL) to the simulator.
If agents complete tasks with expert waypoints, the data is correct.
"""
import json
import os
import sys
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vla.waypoint_tracker import WaypointTracker
from vla.renderer import SimulatorRenderer


class ExpertReplayController:
    """
    Replaces VLA model: reads expert waypoints from collected data
    and feeds them to the waypoint tracker.
    """

    def __init__(self, agents, b2_objects, config):
        self.agents = agents
        self.b2_objects = b2_objects
        self.chunk_size = config.get("chunk_size", 8)
        self.steps_per_sec = config.get("steps_per_sec", 60)
        self.dt = 1.0 / self.steps_per_sec
        self.collect_data = config.get("collect_data", False)

        self.trackers = {}
        for agent in agents:
            self.trackers[agent.id] = WaypointTracker(agent, self.dt)

        self._step_counter = 0
        self._last_inference = -999
        self._expert_data = self._load_data(config.get("data_file", ""))
        self._data_idx = 0

    def _load_data(self, data_file):
        if not data_file or not os.path.exists(data_file):
            return []
        records = []
        with open(data_file) as f:
            for line in f:
                records.append(json.loads(line))
        print(f"Loaded {len(records)} expert records from {data_file}")
        return records

    def step(self, simulator):
        self._step_counter += 1

        needs_inference = (self._step_counter - self._last_inference) >= self.chunk_size
        needs_inference |= any(not t.has_waypoints() for t in self.trackers.values())

        if needs_inference and self._data_idx < len(self._expert_data):
            record = self._expert_data[self._data_idx]
            self._data_idx += 1
            waypoints = {}

            # Get agent order from serializer (matches the collection order)
            from vla.state_serializer import StateSerializer
            _, features, agents_data = StateSerializer().serialize(simulator)

            for ag in agents_data:
                aid = ag["id"]
                key = str(aid)
                if key in record["target_action"]:
                    wps = record["target_action"][key]
                    waypoints[aid] = wps

            self._dispatch_waypoints(waypoints, agents_data)
            self._last_inference = self._step_counter

        for agent in self.agents:
            tracker = self.trackers.get(agent.id)
            if tracker is None or not tracker.has_waypoints():
                continue
            position = agent.position
            if hasattr(position, 'x'):
                from geometry import Point
                position = Point(position.x, position.y)
            vx, vy = tracker.compute_velocity(position)
            agent.linear_velocity = (vx, vy)

    def _dispatch_waypoints(self, waypoints, agents_data):
        for ag in agents_data:
            aid = ag["id"]
            wps = waypoints.get(aid, [])
            if not wps:
                continue
            global_wps = [(wx, wy) for wx, wy in wps]
            self.trackers[aid].set_waypoints(global_wps)

    def flush_data(self):
        print(f"Replay done. Used {self._data_idx}/{len(self._expert_data)} records.")
