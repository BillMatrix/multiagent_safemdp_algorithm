from __future__ import division, print_function, absolute_import

import time
import argparse
from multiprocessing import Pool, cpu_count

import GPy
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle as pkl

from mars_utilities import mars_map
from helper import plot_altitudes, init_dummy_xy, init_dummy_xy_explore, value_iteration
from multiagent_safemdp_agent import MultiagentSafeMDPAgent
from singleagent_safemdp_agent import SingleagentSafeMDPAgent
from epsilon_greedy_agent import EpsilonGreedyAgent

mars_world_shape = (10, 10)
toy_world_shape = (8, 8)
step_size = (1., 1.)
beta = 2
noise = 0.001
epochs = 10
maximum_distance = 5.

def init_multi_safe_agents(num_multi_safe_agents, num_agents, init_x, init_y,
                            init_x_explore, init_y_explore, S0, h, c,
                            world_shape, step_size, max_dist):
    agents = []
    for i in range(num_multi_safe_agents):
        self_reward_kernel = GPy.kern.RBF(input_dim=2, lengthscale=(2., 2.), variance=1., ARD=True)
        self_reward_lik = GPy.likelihoods.Gaussian(variance=noise ** 2)
        self_reward_lik.constrain_bounded(1e-6, 10000.)
        self_gp = GPy.core.GP(init_x, init_y, self_reward_kernel, self_reward_lik)

        others_reward_gps = []
        others_explore_gps = []

        for j in range(num_agents - 1):
            reward_kernel = GPy.kern.RBF(input_dim=2, lengthscale=(2., 2.), variance=1., ARD=True)
            reward_lik = GPy.likelihoods.Gaussian(variance=noise ** 2)
            reward_lik.constrain_bounded(1e-6, 10000.)
            reward_gp = GPy.core.GP(init_x, init_y, reward_kernel, reward_lik)
            others_reward_gps += [reward_gp]

            explore_kernel = GPy.kern.RBF(input_dim=4, lengthscale=(2., 2., 2., 2.), ARD=True)
            explore_lik = GPy.likelihoods.Gaussian(variance=noise ** 2)
            explore_lik.constrain_bounded(1e-6, 10000.)
            explore_gp = GPy.core.GP(init_x_explore, init_y_explore, explore_kernel, explore_lik)
            others_explore_gps += [explore_gp]

        agents += [MultiagentSafeMDPAgent(
            i,
            self_gp,
            others_explore_gps,
            others_reward_gps,
            world_shape,
            step_size,
            beta,
            h,
            c,
            S0,
            S0[i],
            S0[:i] + S0[i + 1:],
            num_agents,
            maximum_distance=max_dist
        )]

    return agents

def init_single_safe_agents(num_multi_safe_agents, num_single_safe_agents, num_agents,
                            init_x, init_y, S0, h, c, world_shape, step_size, max_dist):
    agents = []
    for i in range(num_single_safe_agents):
        self_reward_kernel = GPy.kern.RBF(input_dim=2, lengthscale=(2., 2.), variance=1., ARD=True)
        self_reward_lik = GPy.likelihoods.Gaussian(variance=noise ** 2)
        self_reward_lik.constrain_bounded(1e-6, 10000.)
        self_gp = GPy.core.GP(init_x, init_y, self_reward_kernel, self_reward_lik)

        agents += [SingleagentSafeMDPAgent(
            i + num_multi_safe_agents,
            self_gp,
            world_shape,
            step_size,
            beta,
            h,
            c,
            S0,
            S0[i + num_multi_safe_agents],
            S0[:i + num_multi_safe_agents] + S0[i + num_multi_safe_agents + 1:],
            num_agents,
            maximum_distance=max_dist
        )]

    return agents

def init_epsilon_greedy_agents(num_multi_safe_agents, num_single_safe_agents, num_epsilon_greedy_agents,
                            num_agents, true_value_function, S0, h, c, world_shape, step_size, epsilon, max_dist):
    agents = []
    prev_num_agents = num_multi_safe_agents + num_single_safe_agents
    for i in range(num_epsilon_greedy_agents):
        agents += [EpsilonGreedyAgent(
            i + prev_num_agents,
            world_shape,
            step_size,
            h,
            c,
            S0,
            S0[i + prev_num_agents],
            S0[:i + prev_num_agents] + S0[i + prev_num_agents + 1:],
            num_agents,
            true_value_function,
            epsilon,
            maximum_distance=max_dist
        )]

    return agents

def update_agents(agent):
    agent.update_others_act(agent.new_act)
    agent.update_others_pos(agent.new_pos)
    agent.update_others_reward(agent.new_rewards)
    return agent

def main(args):
    if args.map == 'mars':
        world_shape = mars_world_shape
        altitudes = mars_map(world_shape)
    else:
        world_shape = toy_world_shape
        altitudes = pkl.load(open('altitudes.pkl', 'rb'))
        altitudes = altitudes.reshape(world_shape)

    init_x, init_y = init_dummy_xy(world_shape, step_size, altitudes)
    init_x_explore, init_y_explore = init_dummy_xy_explore(world_shape, step_size)
    max_dist = 0.
    if args.collab == '0':
        num_multi_safe_agents = int(args.num_multi_safe_agents)
        num_single_safe_agents = int(args.num_single_safe_agents)
        num_epsilon_greedy_agents = int(args.num_epsilon_greedy_agents)
    elif args.collab == '1':
        num_multi_safe_agents = 2
        num_single_safe_agents = 0
        num_epsilon_greedy_agents = 0
        max_dist = maximum_distance
    elif args.collab == '2':
        num_multi_safe_agents = 0
        num_single_safe_agents = 0
        num_epsilon_greedy_agents = 2
        max_dist = maximum_distance

    print('Config: We have {0} Multiagent SafeMDP agents, {1} Iterative SafeMDP agents and {2} e-greedy agents'.format(
        num_multi_safe_agents,
        num_single_safe_agents,
        num_epsilon_greedy_agents,
    ))
    num_agents = num_multi_safe_agents + num_single_safe_agents + num_epsilon_greedy_agents
    h = float(args.h)
    c = float(args.c)
    epsilon = 0.05
    true_value_function = value_iteration(altitudes, world_shape, step_size)

    S = []
    for i in range(world_shape[0]):
        for j in range(world_shape[1]):
            if altitudes[i, j] > h:
                S += [(i, j)]

    if len(S) < num_agents:
        print('Invalid domain to set {0} agents'.format(num_agents))
        return

    multi_safe_agents_unsafe = np.array([0 for _ in range(epochs)])
    multi_safe_agents_joint_unsafe = np.array([0 for _ in range(epochs)])
    single_safe_agents_unsafe = np.array([0 for _ in range(epochs)])
    single_safe_agents_joint_unsafe = np.array([0 for _ in range(epochs)])
    epsilon_greedy_agents_unsafe = np.array([0 for _ in range(epochs)])
    epsilon_greedy_agents_joint_unsafe = np.array([0 for _ in range(epochs)])
    collaborative_joint_unsafe = np.array([0 for _ in range(epochs)])

    for ep in range(epochs):
        print(ep)
        if args.collab == '0':
            np.random.shuffle(S)
            S0 = S[:num_agents]
        else:
            S0 = S[:num_agents]

        multi_safe_agents = init_multi_safe_agents(
            num_multi_safe_agents,
            num_agents,
            init_x,
            init_y,
            init_x_explore,
            init_y_explore,
            S0,
            h,
            c,
            world_shape,
            step_size,
            max_dist
        )
        single_safe_agents = init_single_safe_agents(
            num_multi_safe_agents,
            num_single_safe_agents,
            num_agents,
            init_x,
            init_y,
            S0,
            h,
            c,
            world_shape,
            step_size,
            max_dist
        )
        epsilon_greedy_agents = init_epsilon_greedy_agents(
            num_multi_safe_agents,
            num_single_safe_agents,
            num_epsilon_greedy_agents,
            num_agents,
            true_value_function,
            S0,
            h,
            c,
            world_shape,
            step_size,
            epsilon,
            max_dist
        )
        agents = multi_safe_agents + single_safe_agents + epsilon_greedy_agents

        for t in tqdm(range(50)):
            new_pos = []
            new_act = []
            new_rewards = []
            for agent in range(num_agents):
                next_a, next_s = agents[agent].choose_action()
                new_pos += [next_s]
                new_act += [next_a]

                reward = altitudes[int(next_s[0] / step_size[0]), int(next_s[1] / step_size[1])]
                new_rewards += [reward]
                agents[agent].update_self_reward_observation(reward)

            if args.multi == '1':
                for agent in range(num_agents):
                    agents[agent].new_act = new_act[:agent] + new_act[agent + 1:]
                    agents[agent].new_pos = new_pos[:agent] + new_pos[agent + 1:]
                    agents[agent].new_rewards = new_rewards[:agent] + new_rewards[agent + 1:]

                while True:
                    try:
                        p = Pool(cpu_count())
                        agents = p.map(update_agents, agents)
                        break
                    except:
                        pass
            else:
                for agent in range(num_agents):
                    agents[agent].update_others_act(new_act[:agent] + new_act[agent + 1:])
                    agents[agent].update_others_pos(new_pos[:agent] + new_pos[agent + 1:])
                    agents[agent].update_others_reward(new_rewards[:agent] + new_rewards[agent + 1:])

            if args.collab != '0':
                if np.linalg.norm(new_pos[0] - new_pos[1]) > max_dist \
                    or np.linalg.norm(new_pos[0] - new_pos[1]) <= max_dist - 1:
                    collaborative_joint_unsafe[ep] += 1

        for agent in range(num_agents):
            if agent < num_multi_safe_agents:
                multi_safe_agents_unsafe[ep] += agents[agent].num_unsafe
                multi_safe_agents_joint_unsafe[ep] += agents[agent].num_joint_unsafe
            elif agent >= num_multi_safe_agents and agent < num_multi_safe_agents + num_single_safe_agents:
                single_safe_agents_unsafe[ep] += agents[agent].num_unsafe
                single_safe_agents_joint_unsafe[ep] += agents[agent].num_joint_unsafe
            elif agent >= num_multi_safe_agents + num_single_safe_agents:
                epsilon_greedy_agents_unsafe[ep] += agents[agent].num_unsafe
                epsilon_greedy_agents_joint_unsafe[ep] += agents[agent].num_joint_unsafe

    if num_multi_safe_agents != 0:
        multi_safe_agents_unsafe = multi_safe_agents_unsafe / num_multi_safe_agents
        multi_safe_agents_joint_unsafe = multi_safe_agents_joint_unsafe / num_multi_safe_agents

    if num_single_safe_agents != 0:
        single_safe_agents_unsafe = single_safe_agents_unsafe / num_single_safe_agents
        single_safe_agents_joint_unsafe = single_safe_agents_joint_unsafe / num_single_safe_agents

    if num_epsilon_greedy_agents != 0:
        epsilon_greedy_agents_unsafe = epsilon_greedy_agents_unsafe / num_epsilon_greedy_agents
        epsilon_greedy_agents_joint_unsafe = epsilon_greedy_agents_joint_unsafe / num_epsilon_greedy_agents

    if num_multi_safe_agents != 0:
        print('Multi-agent Joint Unsafe States:')
        print(multi_safe_agents_joint_unsafe)
        print('Multi-agent Unsafe States:')
        print(multi_safe_agents_unsafe)

    if num_single_safe_agents != 0:
        print('Iterative SafeMDP Joint Unsafe States:')
        print(single_safe_agents_joint_unsafe)
        print('Iterative SafeMDP Unsafe States:')
        print(single_safe_agents_unsafe)

    if num_epsilon_greedy_agents != 0:
        print('Epsilon Greedy Joint Unsafe States:')
        print(epsilon_greedy_agents_joint_unsafe)
        print('Epsilon Greedy Unsafe States:')
        print(epsilon_greedy_agents_unsafe)

    if args.collab != '0':
        print('Collaborative Joint Unsafe States')
        print(collaborative_joint_unsafe)

    for i in range(world_shape[0]):
        for j in range(world_shape[1]):
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('map', help='enter either random or mars')
    parser.add_argument('num_multi_safe_agents', help='number of multiagent safe algorithm agents')
    parser.add_argument('num_single_safe_agents', help='number of iterative safeMDP agents')
    parser.add_argument('num_epsilon_greedy_agents', help='number of epsilon greedy agents')
    parser.add_argument('collab', help='0 for non-collaborative, 1 for multi-collaborative, 2 for e-greedy-collaborative')
    parser.add_argument('h', help='safety threshold')
    parser.add_argument('c', help='joint safety threshold')
    parser.add_argument('multi', help='0 for no multiprocessing, 1 for multiprocessing')
    args = parser.parse_args()
    main(args)
