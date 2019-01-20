import matplotlib.pyplot as plt
import numpy as np

action_move_dict = {
    0: np.array([0, 0]),
    1: np.array([-1, 0]),
    2: np.array([0, -1]),
    3: np.array([1, 0]),
    4: np.array([0, 1]),
}

reverse_action_dict = {
    0: 0,
    1: 2,
    2: 1,
    3: 4,
    4: 3,
}

def move_coordinate(start_coord, action, world_shape, step_size):
    new_coord = start_coord + action_move_dict[action] * step_size
    if new_coord[0] < 0.0:
        new_coord[0] = step_size[0] * (world_shape[0] - 1)
    if new_coord[0] > step_size[0] * (world_shape[0] - 1):
        new_coord[0] = 0.0
    if new_coord[1] < 0.0:
        new_coord[1] = step_size[1] * (world_shape[1] - 1)
    if new_coord[1] > step_size[1] * (world_shape[1] - 1):
        new_coord[1] = 0.0

    return new_coord

def plot_altitudes(altitudes, title):
    plt.imshow(altitudes.T, origin="lower", interpolation="nearest")
    plt.colorbar()
    plt.title(title)
    plt.show()

def init_dummy_xy(world_shape, step_size, altitudes):
    i = np.random.choice(world_shape[0], 1)[0]
    j = np.random.choice(world_shape[1], 1)[0]
    coord = np.array([[i * step_size[0], j * step_size[1]]])
    dummy_y = np.array([[altitudes[i, j]]]) + np.random.randn(1, 1)

    return coord, dummy_y

def init_dummy_xy_explore(world_shape, step_size):
    return np.array([[0., 0., 0., 0.]]), np.array([[0.]])

def value_iteration(altitudes, world_shape, step_size):
    difference = 10000.0
    value_functions = np.zeros(world_shape)
    while difference > 0.01:
        cur_difference = 0.0
        for i in range(world_shape[0]):
            for j in range(world_shape[1]):
                cur_reward = altitudes[i, j]
                old_value = value_functions[i, j]
                for action in range(0, 5):
                    new_state = move_coordinate(
                        np.array([i * step_size[0], j * step_size[1]]),
                        action,
                        world_shape,
                        step_size
                    )
                    new_i = int(new_state[0] / step_size[0])
                    new_j = int(new_state[1] / step_size[1])
                    value_functions[i, j] = max(
                        cur_reward + 0.9 * value_functions[new_i, new_j],
                        old_value
                    )

                cur_difference = max(
                    cur_difference,
                    abs(value_functions[i, j] - old_value)
                )

        difference = cur_difference

    return value_functions
