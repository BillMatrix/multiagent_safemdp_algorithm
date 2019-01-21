from __future__ import division, print_function, absolute_import
import operator

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

from helper import action_move_dict, reverse_action_dict, move_coordinate

class EpsilonGreedyAgent():
    def __init__(self, index, world_shape, step_size, h, c, S0, my_pos,
                others_pos, num_agents, value_functions, epsilon):
        self.index = index
        self.S = S0.copy()
        self.world_shape = world_shape
        self.step_size = step_size
        self.h = h
        self.c = c
        self.my_pos = my_pos
        self.others_pos = others_pos
        self.num_other_agents = num_agents - 1
        self.value_functions = value_functions

        self.other_agent_vacancy = np.zeros(world_shape)
        self.other_agent_vacancy[:, :] = 1.

        self.num_unsafe = 0
        self.num_joint_unsafe = 0
        self.epsilon = epsilon

    def choose_action(self):
        for agent in range(self.num_other_agents):
            agent_coord = self.others_pos[agent]

            possible_coords = []
            for action in range(5):
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
        for action in range(5):
            next_coord = move_coordinate(self.my_pos, action, self.world_shape, self.step_size)
            self_possible_coords += [next_coord]
            prob_vacancy = self.other_agent_vacancy[
                int(next_coord[0] / self.step_size[0]),
                int(next_coord[1] / self.step_size[1])
            ]
            self_possible_vacancy += [prob_vacancy]
            lr = self.value_functions[
                int(next_coord[0] / self.step_size[0]),
                int(next_coord[1] / self.step_size[1])
            ]
            self_possible_safety += [lr]
            if prob_vacancy > self.c:
                self_possible_actions += [action]

        target_coord = self.my_pos
        target_action = 0
        max_value = 0

        if len(self_possible_actions) == 0:
            action, value = max(enumerate(self_possible_safety), key=operator.itemgetter(1))
            self.my_pos = self_possible_coords[action]
            target_action, target_coord = action, self_possible_coords[action]

        for action in self_possible_actions:
            if self_possible_safety[action] > max_value:
                max_value = self_possible_safety[action]
                target_action = action
                target_coord = self_possible_coords[action]

        if np.random.random_sample() < self.epsilon:
            target_action = np.random.choice(5, 1)[0]
            target_coord = possible_coords[target_action]

        self.my_pos = target_coord
        return target_action, target_coord

    def update_self_reward_observation(self, reward):
        if reward < self.h:
            self.num_unsafe += 1
        return

    def update_others_pos(self, positions):
        self.others_pos = positions
        for agent, pos in enumerate(positions):
            if self.my_pos[0] == pos[0] and self.my_pos[1] == pos[1]:
                self.num_joint_unsafe += 1

    def update_others_act(self, actions):
        return

    def update_others_reward(self, rewards):
        return
