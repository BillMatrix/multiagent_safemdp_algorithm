from __future__ import division, print_function, absolute_import
import operator

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

from helper import action_move_dict, reverse_action_dict, move_coordinate

class SingleagentSafeMDPAgent():
    def __init__(self, index, self_rewards_gp, world_shape, step_size, beta, h, c, S0, my_pos,
                others_pos, num_agents, maximum_distance=0, gamma=0.9):
        self.index = index
        self.S = S0.copy()
        self.rewards_gp = self_rewards_gp
        self.world_shape = world_shape
        self.step_size = step_size
        self.beta = beta
        self.h = h
        self.c = c
        self.gamma = gamma
        self.my_pos = my_pos
        self.my_traj = []
        self.my_rewards = []
        self.others_pos = others_pos
        self.num_other_agents = num_agents - 1
        self.maximum_distance = maximum_distance
        if self.maximum_distance != 0:
            self.c = -self.c

        self.self_l = np.zeros(world_shape)
        self.self_u = np.zeros(world_shape)
        self.self_l[:, :] = -100.0
        self.self_u[:, :] = 100.0
        for s in S0:
            self.self_l[s[0], s[1]] = h

        self.other_agent_vacancy = np.zeros(world_shape)
        self.other_agent_vacancy[:, :] = 1.

        self.num_unsafe = 0
        self.num_joint_unsafe = 0
        self.coords_visited = []
        self.new_reward = 0
        self.new_act = []
        self.new_pos = []
        self.new_rewards = []

    def choose_action(self):
        for agent in range(self.num_other_agents):
            agent_coord = self.others_pos[agent]

            possible_coords = []
            for action in range(1, 5):
                possible_coords += [move_coordinate(agent_coord, action, self.world_shape, self.step_size)]

            for coord in possible_coords:
                self.other_agent_vacancy[
                    int(coord[0] / self.step_size[0]),
                    int(coord[1] / self.step_size[1])
                ] *= 0.8

        self_possible_coords = []
        self_possible_vacancy = []
        self_possible_safety = []
        self_possible_actions = []
        for action in range(1, 5):
            next_coord = move_coordinate(self.my_pos, action, self.world_shape, self.step_size)
            self_possible_coords += [next_coord]
            prob_vacancy = 0.0
            if self.maximum_distance == 0:
                prob_vacancy = self.other_agent_vacancy[
                    int(next_coord[0] / self.step_size[0]),
                    int(next_coord[1] / self.step_size[1])
                ]
            else:
                for i in range(self.world_shape[0]):
                    for j in range(self.world_shape[1]):
                        cur_coord = np.array([i * self.step_size[0], j * self.step_size[1]])
                        if self.other_agent_vacancy[i, j] != 1.0 \
                                and np.linalg.norm(cur_coord - next_coord) <= self.maximum_distance \
                                and np.linalg.norm(cur_coord - next_coord) > self.maximum_distance - 1:
                            prob_vacancy = max(prob_vacancy, self.other_agent_vacancy[i, j])
                prob_vacancy = -prob_vacancy

            self_possible_vacancy += [prob_vacancy]
            lr = self.self_l[
                int(next_coord[0] / self.step_size[0]),
                int(next_coord[1] / self.step_size[1])
            ]
            self_possible_safety += [lr]
            visited = False
            for c in self.my_traj:
                if c[0] == next_coord[0] and c[1] == next_coord[1]:
                    visited = True
            if lr > self.h and prob_vacancy > self.c and self._returnable(next_coord) and not visited:
                self_possible_actions += [action]

        if len(self_possible_actions) == 0:
            action, value = max(enumerate(self_possible_safety), key=operator.itemgetter(1))
            target_coord = self_possible_coords[action - 1]
            self.my_pos = target_coord
            visited = False
            for c in self.coords_visited:
                if target_coord[0] == c[0] and target_coord[1] == c[1]:
                    visited = True

            if not visited:
                self.coords_visited += [target_coord]
            self.my_traj += [target_coord]
            return action, target_coord

        target_coord = self.my_pos
        target_action = 0
        max_conf_interval = 0
        for action in self_possible_actions:
            ub = self.self_u[
                int(self_possible_coords[action - 1][0] / self.step_size[0]),
                int(self_possible_coords[action - 1][1] / self.step_size[1])
            ]
            lb = self.self_l[
                int(self_possible_coords[action - 1][0] / self.step_size[0]),
                int(self_possible_coords[action - 1][1] / self.step_size[1])
            ]
            conf_interval = ub - lb
            if conf_interval > max_conf_interval:
                max_conf_interval = conf_interval
                target_action = action
                target_coord = self_possible_coords[action - 1]

        self.my_pos = target_coord
        visited = False
        for c in self.coords_visited:
            if target_coord[0] == c[0] and target_coord[1] == c[1]:
                visited = True

        if not visited:
            self.coords_visited += [target_coord]
        self.my_traj += [target_coord]
        return target_action, target_coord

    def update_self_reward_observation(self, reward):
        self.my_rewards += [[reward]]
        # print(self.my_traj)
        # print(self.my_rewards)
        self.rewards_gp.set_XY(
            np.array(self.my_traj),
            np.array(self.my_rewards),
        )
        if reward < self.h:
            self.num_unsafe += 1

        for i in range(self.world_shape[0]):
            for j in range(self.world_shape[1]):
                new_reward, new_var = self.rewards_gp.predict(
                    np.array([[i * self.step_size[0], j * self.step_size[1]]]),
                    kern=self.rewards_gp.kern,
                    full_cov=False
                )
                new_u = new_reward + self.beta * np.sqrt(new_var)
                new_l = new_reward - self.beta * np.sqrt(new_var)
                self.self_l[i, j] = max(self.self_l[i, j], new_l)
                self.self_u[i, j] = min(self.self_u[i, j], new_u)

    def update_others_pos(self, positions):
        self.others_pos = positions
        for _, pos in enumerate(positions):
            if self.my_pos[0] == pos[0] and self.my_pos[1] == pos[1]:
                self.num_joint_unsafe += 1

    def update_others_act(self, actions):
        return

    def update_others_reward(self, rewards):
        return

    def _returnable(self, coord):
        for action in range(1, 5):
            new_coord = move_coordinate(coord, action, self.world_shape, self.step_size)
            if self.self_l[
                int(new_coord[0] / self.step_size[0]),
                int(new_coord[1] / self.step_size[1])
            ] > self.h:
                return True

        return False
