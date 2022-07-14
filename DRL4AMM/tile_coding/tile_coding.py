# https://towardsdatascience.com/reinforcement-learning-tile-coding-implementation-7974b600762b
# https://github.com/MJeremy2017/Reinforcement-Learning-Implementation/blob/master/TileCoding/tile_coding.py

import numpy as np


def create_tiling(feat_range, bins, offset):
    """
    Create 1 tiling spec of 1 dimension(feature)
    feat_range: feature range; example: [-1, 1]
    bins: number of bins for that feature; example: 10
    offset: offset for that feature; example: 0.2
    """

    return np.linspace(feat_range[0], feat_range[1], bins + 1)[1:-1] + offset


def create_tilings(feature_ranges, number_tilings, bins, offsets):
    """
    feature_ranges: range of each feature; example: x: [-1, 1], y: [2, 5] -> [[-1, 1], [2, 5]]
    number_tilings: number of tilings; example: 3 tilings
    bins: bin size for each tiling and dimension; example: [[10, 10], [10, 10], [10, 10]]: 3 tilings * [x_bin, y_bin]
    offsets: offset for each tiling and dimension; example: [[0, 0], [0.2, 1], [0.4, 1.5]]: 3 tilings * [x_offset, y_offset]
    """
    tilings = []
    # for each tiling
    for tile_i in range(number_tilings):
        tiling_bin = bins[tile_i]
        tiling_offset = offsets[tile_i]

        tiling = []
        # for each feature dimension
        for feat_i in range(len(feature_ranges)):
            feat_range = feature_ranges[feat_i]
            # tiling for 1 feature
            feat_tiling = create_tiling(feat_range, tiling_bin[feat_i], tiling_offset[feat_i])
            tiling.append(feat_tiling)
        tilings.append(tiling)
    return np.array(tilings)


def get_tile_coding(feature, tilings):
    """
    feature: sample feature with multiple dimensions that need to be encoded; example: [0.1, 2.5], [-0.3, 2.0]
    tilings: tilings with a few layers
    return: the encoding for the feature on each layer
    """
    num_dims = len(feature)
    feat_codings = []
    for tiling in tilings:
        feat_coding = []
        for i in range(num_dims):
            feat_i = feature[i]
            tiling_i = tiling[i]  # tiling on that dimension
            coding_i = np.digitize(feat_i, tiling_i)
            feat_coding.append(coding_i)
        feat_codings.append(feat_coding)
    return np.array(feat_codings)


# example Q-function


class QValueFunction:
    def __init__(self, tilings, actions, lr, gamma, eps):
        self.tilings = tilings
        self.num_tilings = len(self.tilings)
        self.actions = actions
        self.lr = lr  # /self.num_tilings  # learning rate equally assigned to each tiling
        self.state_sizes = [
            tuple(len(splits) + 1 for splits in tiling) for tiling in self.tilings
        ]  # [(10, 10), (10, 10), (10, 10)]
        self.q_tables = [np.zeros(shape=(state_size + (len(self.actions),))) for state_size in self.state_sizes]

        # added by rahul 9-June-2022
        self.gamma = gamma
        self.eps = eps

    def value(self, state, action):
        state_codings = get_tile_coding(state, self.tilings)  # [[5, 1], [4, 0], [3, 0]] ...
        action_idx = self.actions.index(action)

        value = 0
        for coding, q_table in zip(state_codings, self.q_tables):
            # for each q table
            value += q_table[tuple(coding) + (action_idx,)]
        return value / self.num_tilings

    def update(self, state, action, target):
        state_codings = get_tile_coding(state, self.tilings)  # [[5, 1], [4, 0], [3, 0]] ...
        action_idx = self.actions.index(action)

        for coding, q_table in zip(state_codings, self.q_tables):
            delta = target - q_table[tuple(coding) + (action_idx,)]
            q_table[tuple(coding) + (action_idx,)] += self.lr * (delta)

    ###########################################################################
    # added by rahul 9-June-2022
    ###########################################################################
    def get_action_q_value_dict(self, state):
        return {action: self.value(state, action) for action in self.actions}

    def get_best_q_value(self, state):
        """
        max_{a in actions} Q(state, a)
        """
        return max(self.get_action_q_value_dict(state).values())

    def greedy(self, state):
        """
        returns an action that maximises the q value estimate for the given state
        """

        q_value_dict = self.get_action_q_value_dict(state)

        best_value = max(q_value_dict.values())

        best_actions = [action for action in self.actions if q_value_dict[action] == best_value]

        return best_actions[0]

    def eps_greedy(self, state):

        if np.random.rand(1) < self.eps:
            return self.actions[np.random.choice(range(len(self.actions)))]

        else:
            return self.greedy(state)

    def get_target(self, reward, new_state):
        """
        target = reward_t + discount_factor * [ max_{a in actions} Q(new_state, a) ]
        """
        return reward + self.gamma * self.get_best_q_value(new_state)
