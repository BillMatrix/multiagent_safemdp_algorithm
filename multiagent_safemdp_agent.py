from __future__ import division, print_function, absolute_import
import operator

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

from helper import action_move_dict, reverse_action_dict, move_coordinate

class MultiagentSafeMDPAgent():
    def __init__(self, self_rewards_gp, others_explore_gp, others_rewards_gp,
                world_shape, step_size, beta, h, c, S0, my_pos,
                others_pos, num_agents, gamma=0.9):

        self.S = S0.copy()
        self.rewards_gp = self_rewards_gp
        self.world_shape = world_shape
        self.step_size = step_size
        self.others_explore_gp = others_explore_gp
        self.others_rewards_gp = others_rewards_gp
        self.beta = beta
        self.h = h
        self.c = c
        self.gamma = gamma
        self.my_pos = my_pos
        self.others_pos = others_pos
        self.num_other_agents = num_agents - 1

        self.self_l = np.zeros(world_shape)
        self.self_u = np.zeros(world_shape)
        self.self_l[:, :] = -100.0
        self.self_u[:, :] = 100.0
        for s in S0:
            self.self_l[s[0], s[1]] = h

        self.others_l = np.zeros((num_agents - 1, world_shape[0], world_shape[1]))
        self.others_u = np.zeros((num_agents - 1, world_shape[0], world_shape[1]))
        self.others_l[:, :, :] = -100.0
        self.others_u[:, :, :] = 100.0
        self.others_value_func = np.zeros((num_agents - 1, world_shape[0], world_shape[1]))
        self.others_value_func_l = np.zeros((num_agents - 1, world_shape[0], world_shape[1]))
        self.others_value_func_u = np.zeros((num_agents - 1, world_shape[0], world_shape[1]))
        self.others_value_func_l[:, :, :] = -100.0
        self.others_value_func_u[:, :, :] = 100.0
        for s in S0:
            self.others_l[:, s[0], s[1]] = h
        self.epsilons = [0. for i in range(self.num_other_agents)]

        for agent in range(self.num_other_agents):
            self._conf_value_iteration(agent)

        self.other_agent_vacancy = np.zeros(world_shape)
        self.other_agent_vacancy[:, :] = 1.

        self.trajs = [[self.others_pos[agent]] for agent in range(self.num_other_agents)]
        self.num_unsafe = 0
        self.num_joint_unsafe = 0
        self.coords_visited = []

    def choose_action(self):
        for agent in range(self.num_other_agents):
            agent_coord = self.others_pos[agent]
            epsilon = self.epsilons[agent]

            possible_coords = []
            for action in range(1, 5):
                possible_coords += [move_coordinate(agent_coord, action, self.world_shape, self.step_size)]

            explore_gp_pred = self._get_explore_gp_pred(agent, agent_coord, possible_coords)

            for i, coord in enumerate(possible_coords):
                prob_action_exploit = self._get_exploit_upper_bound(agent, coord, possible_coords)
                prob_action_explore = self._get_explore_upper_bound(agent, i, explore_gp_pred)
                prob_action = epsilon * prob_action_explore + (1 - epsilon) * prob_action_exploit

                self.other_agent_vacancy[
                    int(coord[0] / self.step_size[0]),
                    int(coord[1] / self.step_size[1])
                ] *= (1 - prob_action)

        self_possible_coords = []
        self_possible_vacancy = []
        self_possible_safety = []
        self_possible_actions = []
        for action in range(1, 5):
            next_coord = move_coordinate(self.my_pos, action, self.world_shape, self.step_size)
            self_possible_coords += [next_coord]
            prob_vacancy = self.other_agent_vacancy[
                int(next_coord[0] / self.step_size[0]),
                int(next_coord[1] / self.step_size[1])
            ]
            self_possible_vacancy += [prob_vacancy]
            lr = self.self_l[
                int(next_coord[0] / self.step_size[0]),
                int(next_coord[1] / self.step_size[1])
            ]
            self_possible_safety += [lr]

            if lr > self.h and prob_vacancy > self.c and self._returnable(next_coord):
                self_possible_actions += [action]

        if len(self_possible_actions) == 0:
            action, value = max(enumerate(self_possible_vacancy), key=operator.itemgetter(1))
            target_coord = self_possible_coords[action - 1]
            self.my_pos = target_coord
            visited = False
            for c in self.coords_visited:
                if target_coord[0] == c[0] and target_coord[1] == c[1]:
                    visited = True

            if not visited:
                self.coords_visited += [target_coord]
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
        return target_action, target_coord

    def update_self_reward_observation(self, reward):
        self.rewards_gp.set_XY(
            np.array([self.my_pos]),
            np.array([[reward]]),
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
        for agent, pos in enumerate(positions):
            if self.my_pos[0] == pos[0] and self.my_pos[1] == pos[1]:
                self.num_joint_unsafe += 1
            self.trajs[agent] += [pos]

    def update_others_act(self, actions):
        for agent in range(self.num_other_agents):
            cur_action = actions[agent]
            for action in range(1, 5):
                prev_coord = self.others_pos[agent]
                cur_coord = move_coordinate(prev_coord, action, self.world_shape, self.step_size)
                if action == cur_action:
                    self.others_explore_gp[agent].set_XY(
                        np.array([[prev_coord[0], prev_coord[1], cur_coord[0], cur_coord[1]]]),
                        np.array([[0.]]),
                    )
                else:
                    self.others_explore_gp[agent].set_XY(
                        np.array([[prev_coord[0], prev_coord[1], cur_coord[0], cur_coord[1]]]),
                        np.array([[1.]]),
                    )

    def update_others_reward(self, rewards):
        for agent in range(self.num_other_agents):
            self.others_rewards_gp[agent].set_XY(
                np.array([self.others_pos[agent]]),
                np.array([[rewards[agent]]])
            )
            self._conf_value_iteration(agent)

            self._optimize_for_epsilon(agent)

    def _conf_value_iteration(self, agent):
        diff = 10000

        while diff > 0.01:
            cur_diff = 0.0
            for i in range(self.world_shape[0]):
                for j in range(self.world_shape[1]):
                    x_coord = i * self.step_size[0]
                    y_coord = j * self.step_size[1]
                    state = np.array([x_coord, y_coord])
                    cur_reward, cur_var = self.others_rewards_gp[agent].predict(
                        np.array([state]),
                        kern=self.others_rewards_gp[agent].kern,
                        full_cov=False
                    )
                    cur_value = self.others_value_func[agent, i, j]
                    best_next_state = state
                    for action in range(1, 5):
                        new_state = move_coordinate(state, action, self.world_shape, self.step_size)
                        new_i = int(new_state[0] / self.step_size[0])
                        new_j = int(new_state[1] / self.step_size[1])
                        new_value = cur_reward + self.gamma * self.others_value_func[agent, new_i, new_j]
                        if new_value > cur_value:
                            best_next_state = new_state
                            self.others_value_func[agent, i, j] = new_value

                    new_i = int(best_next_state[0] / self.step_size[0])
                    new_j = int(best_next_state[1] / self.step_size[1])
                    next_value_var = self.others_value_func[agent, new_i, new_j] \
                        - self.others_value_func_l[agent, new_i, new_j]
                    std_dev = np.sqrt(self.beta ** 2 * cur_var + self.gamma ** 2 * next_value_var)

                    self.others_value_func_l[agent, i, j] = max(
                        self.others_value_func_l[agent, i, j],
                        self.others_value_func[agent, i, j] - std_dev
                    )
                    self.others_value_func_u[agent, i, j] = min(
                        self.others_value_func_u[agent, i, j],
                        self.others_value_func[agent, i, j] + std_dev
                    )

                    cur_diff = max(
                        cur_diff,
                        abs(self.others_value_func[agent, i, j] - cur_value)
                    )

            diff = cur_diff

    def _get_exploit_upper_bound(self, agent, coord, possible_coords):
        denominator_prefix = sum(
            [np.exp(self.others_value_func_l[
                agent,
                int(c[0] / self.step_size[0]),
                int(c[1] / self.step_size[1])
            ])
            for c in possible_coords if c[0] != coord[0] or c[1] != coord[1]]
        )
        nominator = np.exp(
            self.others_value_func_u[
                agent,
                int(coord[0] / self.step_size[0]),
                int(coord[1] / self.step_size[1])
            ])
        denominator = nominator + denominator_prefix
        prob_action_exploit = nominator / denominator
        return prob_action_exploit

    def _get_explore_upper_bound(self, agent, i, explore_gp_pred):
        denominator_prefix = sum(
            [np.exp(explore_gp_pred[item][0] - np.sqrt(explore_gp_pred[item][1]))
            for item in range(len(explore_gp_pred)) if item != i]
        )
        nominator = np.exp(explore_gp_pred[i][0] + np.sqrt(explore_gp_pred[i][1]))
        return nominator / (nominator + denominator_prefix)

    def _get_explore_gp_pred(self, agent, agent_coord, possible_coords):
        return [
            self.others_explore_gp[agent].predict(
                np.array([[agent_coord[0], agent_coord[1], c[0], c[1]]]),
                kern=self.others_explore_gp[agent].kern,
                full_cov=False
            )
            for c in possible_coords
        ]

    def _optimize_for_epsilon(self, agent):
        traj = self.trajs[agent]
        def _compute_log_likelihood(epsilon):
            sum_log_likelihood = 0.0
            for step in range(1, len(traj)):
                prev_coord = traj[step - 1]
                cur_coord = traj[step]
                possible_coords = [
                    move_coordinate(prev_coord, a, self.world_shape, self.step_size)
                    for a in range(1, 5)
                ]
                explore_denominator = sum([
                    np.exp(self.others_explore_gp[agent].predict(
                        np.array([[prev_coord[0], prev_coord[1], c[0], c[1]]]),
                        kern=self.others_explore_gp[agent].kern,
                        full_cov=False
                    )[0])
                    for c in possible_coords
                ])
                explore_nominator = np.exp(self.others_explore_gp[agent].predict(
                    np.array([[prev_coord[0], prev_coord[1], cur_coord[0], cur_coord[1]]]),
                    kern=self.others_explore_gp[agent].kern,
                    full_cov=False
                )[0])
                explore_prob = explore_nominator / explore_denominator

                exploit_denominator = sum([
                    np.exp(self.others_value_func[agent, int(c[0] / self.step_size[0]), int(c[1]/ self.step_size[1])])
                    for c in possible_coords
                ])
                exploit_nominator = np.exp(
                    self.others_value_func[agent, int(cur_coord[0] / self.step_size[0]), int(cur_coord[1] / self.step_size[1])]
                )
                exploit_prob = exploit_nominator / exploit_denominator

                sum_log_likelihood += np.log(epsilon[0] * explore_prob + (1 - epsilon[0]) * exploit_prob)

            return -sum_log_likelihood
        res = minimize(_compute_log_likelihood, np.array([0.5]), method='L-BFGS-B', bounds=np.array([(1e-6, 1.0)]))
        self.epsilons[agent] = res.x[0]

    def _returnable(self, coord):
        for action in range(1, 5):
            new_coord = move_coordinate(coord, action, self.world_shape, self.step_size)
            if self.self_l[
                int(new_coord[0] / self.step_size[0]),
                int(new_coord[1] / self.step_size[1])
            ] > self.h:
                return True

        return False
