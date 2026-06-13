"""
@Copyright Dorabot Inc.
@date : 2018-10
@author: {xiaoyu.ge, chen.chong2}@dorabot.com
@brief : local planner based on virtual forces
"""
from geometry import Vector, compute_direction, Point
from local_planner import LocalPlanner
from math import sqrt
class VirtualForcePlanner(LocalPlanner):
    """

    Keyword arguments:
    position -- agent position
    velocity -- linear velocity of the agent
    sensor_observation -- use perception module to emulate sensor observation
    global_planner_path -- list of poses (way points) obtained from global planner
    Return:
    velcocity -- velocity command [only linear velocity returned in current version]
    local_path -- optional
    """
    def compute_plan(self, position, velocity, gridmap, sensor_observation, global_planner_path):
        local_path = []
        start_pose = position
        
        try:
            goal_pose = [x for x in global_planner_path if x.distance(position)<2][-1]
        except:
            goal_pose = global_planner_path[-1]

        direction = compute_direction(start_pose, goal_pose)

        result_vel = Vector(velocity[0], velocity[1])
        for agent in sensor_observation.other_agents_state_in_range_of(5):
            vel_A_B_dir = compute_direction(position, agent.position).normalize()
            # TODO Gary: @Gary do not modify agent status in hidden functions
            self.agent.potential_collision = True
            force_B_A = self.__combined_force(position, agent.position)
            result_vel = result_vel + force_B_A
        for body in sensor_observation.ports_in_range_of(4):
            self.agent.potential_collision = True
            obstacle=body.userData
            pB=Point(obstacle.location.x+obstacle.dimension[0]*0.5,obstacle.location.y+obstacle.dimension[1]*0.5)
            force_B_A = self.__combined_force(position, pB)
            result_vel=result_vel+force_B_A
            
        force_goal = self.__goal_attraction(position, goal_pose)
        result_vel = result_vel + force_goal
        ratio = sqrt(result_vel.x**2 + result_vel.y **2)
        result_vel = result_vel.scale(self.agent.cruise_speed/ratio)
        return result_vel.to_tuple()
    def __repel_force(self, pos_a, pos_b):
        dist = pos_a.distance(pos_b)
        vec = compute_direction(pos_a, pos_b).normalize()
        force= -20/(dist**4)
        return vec.scale(force)
    def __combined_force(self, pos_a, pos_b):
        return self.__repel_force(pos_a, pos_b)
    def __goal_attraction(self, pos_current, loc_destination):
        dist = pos_current.distance(loc_destination)
        vec = compute_direction(pos_current, loc_destination).normalize()
        return vec.scale(100/(dist+0.01))
