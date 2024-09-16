import gym
import numpy as np
from gym import spaces
import torch
import math


class TradingEnv(gym.Env):
    def __init__(self, data, action_name, gamma, n_step=3, batch_size=50, start_index_reward=0, transaction_cost=0):
        super(TradingEnv, self).__init__()

        self.data = data
        self.states = self.data.values
        self.current_state_index = -1
        self.own_share = False
        self.batch_size = batch_size
        self.n_step = n_step
        self.position = 0
        self.initial_balance = 1000
        self.gamma = gamma
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.close_price = list(self.data['close'])
        self.action_name = action_name
        self.code_to_action = {0: 'buy', 1: 'None', 2: 'sell'}
        self.start_index_reward = start_index_reward
        self.trading_cost_ratio = transaction_cost

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # buy, hold, sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.states[0]),), dtype=np.float32)

        self.state_size = len(self.states[0])

    def reset(self):
        self.current_state_index = -1
        self.own_share = False

    def step(self, action):

        done = False
        next_state = None

        if self.current_state_index + self.n_step < len(self.states):
            next_state = self.states[self.current_state_index + self.n_step]
        else:
            done = True

        if action == 0:
            self.own_share = True
        elif action == 2:
            self.own_share = False

        reward = 0
        if not done:
            reward = self.get_reward(action)

        return done, reward, next_state

    def get_current_state(self):
        self.current_state_index += 1
        if self.current_state_index >= len(self.states):
            return None
        return self.states[self.current_state_index]

    def get_reward(self, action):

        reward_index_first = self.current_state_index + self.start_index_reward
        reward_index_last = self.current_state_index + self.start_index_reward + self.n_step \
            if self.current_state_index + self.n_step < len(self.states) else len(self.close_price) - 1

        p1 = self.close_price[reward_index_first]
        p2 = self.close_price[reward_index_last]

        reward = 0
        if p1 and p2 != 0:
            if action == 0 or (action == 1 and self.own_share):  # Buy Share or Hold Share
                reward = ((1 - self.trading_cost_ratio) ** 2 *
                          p2 / p1 - 1) * 100  # profit in percent
            # Sell Share or No Share
            elif action == 2 or (action == 1 and not self.own_share):
                # consider the transaction in the reverse order
                reward = ((1 - self.trading_cost_ratio)
                          ** 2 * p1 / p2 - 1) * 100

        return reward

    def calculate_reward_for_one_step(self, action, index, rewards):
        index += self.start_index_reward  # Last element inside the window
        if action == 0 or (action == 1 and self.own_share):  # Buy Share or Hold Share
            difference = self.close_price[index + 1] - self.close_price[index]
            rewards.append(difference)

        elif action == 2 or (action == 1 and not self.own_share):  # Sell Share or No Share
            difference = self.close_price[index] - self.close_price[index + 1]
            rewards.append(difference)

    def __iter__(self):
        self.index_batch = 0
        self.num_batch = math.ceil(len(self.states) / self.batch_size)
        return self

    def __next__(self):
        if self.index_batch < self.num_batch:
            batch = [torch.tensor([s], dtype=torch.float, device=self.device) for s in
                     self.states[self.index_batch * self.batch_size: (self.index_batch + 1) * self.batch_size]]
            self.index_batch += 1
            return torch.cat(batch)

        raise StopIteration

    def get_total_reward(self, action_list):
        total_reward = 0
        for a in action_list:
            if a == 0:
                self.own_share = True
            elif a == 2:
                self.own_share = False
            self.current_state_index += 1
            total_reward += self.get_reward(a)

        return total_reward

    def make_investment(self, action_list):
        self.data[self.action_name] = 'None'
        i = self.start_index_reward + 1
        for a in action_list:
            self.data[self.action_name][i] = self.code_to_action[a]
            i += 1
