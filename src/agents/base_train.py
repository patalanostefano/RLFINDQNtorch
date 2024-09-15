from collections import namedtuple
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from src.agents.dql.dqn import DQN
from itertools import count
from tqdm import tqdm
import math
import os

from pathlib import Path
from src.agents.evaluation import Evaluation

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class BaseTrain:
    def __init__(self,
                 data_loader,
                 data_train,
                 data_test,
                 dataset_name,
                 model_type,
                 window_size=1,
                 transaction_cost=0.0,
                 BATCH_SIZE=30,
                 GAMMA=0.7,
                 ReplayMemorySize=50,
                 TARGET_UPDATE=5,
                 n_step=10,
                 windowed=False):

        self.data_train = data_train
        self.data_test = data_test
        self.DATASET_NAME = dataset_name
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.transaction_cost = transaction_cost
        self.model_type = model_type
        self.window_size = window_size
        self.windowed = windowed

        self.split_point = data_loader.split_point
        self.begin_date = data_loader.begin_date
        self.end_date = data_loader.end_date

        self.TARGET_UPDATE = TARGET_UPDATE
        self.n_step = n_step

        if windowed:
            # Calculate the longest trend duration
            longest_trend = 36
            self.ReplayMemorySize = longest_trend
        else:
            self.ReplayMemorySize = ReplayMemorySize

        self.memory = ReplayMemory(self.ReplayMemorySize)

        self.train_test_split = True if data_test is not None else False

        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 0.0000005

        self.steps_done = 0

        self.PATH = os.path.join(Path(os.path.abspath(os.path.dirname(__file__))).parent,
                                 f'Results/{self.DATASET_NAME}/'
                                 f'{self.model_type}; '
                                 f'BEGIN_DATE({self.begin_date}); '
                                 f'END_DATE({self.end_date}); '
                                 f'SPLIT_POINT({self.split_point}); '
                                 f'WindowSize({self.window_size}); '
                                 f'BATCH_SIZE{self.BATCH_SIZE}; '
                                 f'GAMMA{self.GAMMA}; '
                                 f'REPLAY_MEMORY_SIZE{self.ReplayMemorySize}; '
                                 f'TARGET_UPDATE{self.TARGET_UPDATE}; '
                                 f'N_STEP{self.n_step}')

        if not os.path.exists(self.PATH):
            os.makedirs(self.PATH)

        self.model_dir = os.path.join(self.PATH, f'model.pkl')

        self.trend_strength_threshold = 36.0  # Adjust this value as needed

    def is_strong_trend(self, state):
        # Extract trend strength from the state
        # Assuming the trend strength is the last element in the state vector
        trend_strength = state[-1]
        return trend_strength > self.trend_strength_threshold

    def optimize_model_for_trend(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.memory[-self.BATCH_SIZE:]
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net

        # Using policy-net, we calculate the action-value of the previous actions we have taken before.
        state_action_values = self.policy_net(
            state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(
            non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * (self.GAMMA ** self.n_step)) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def select_action(self, state):
        sample = random.random()

        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                self.policy_net.eval()
                action = self.policy_net(state).max(1)[1].view(1, 1)
                self.policy_net.train()
                return action
        else:
            return torch.tensor([[random.randrange(3)]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net

        # Using policy-net, we calculate the action-value of the previous actions we have taken before.
        state_action_values = self.policy_net(
            state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(
            non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * (self.GAMMA ** self.n_step)) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self, num_episodes=50):
        print('Training')
        for i_episode in tqdm(range(num_episodes)):
            # Initialize the environment and state
            self.data_train.reset()
            current_state = self.data_train.get_current_state()

            # Convert to tensor only after checking trend strength
            state = torch.tensor(
                [current_state], dtype=torch.float, device=device)
            for t in count():
                # Select and perform an action
                action = self.select_action(state)
                # print(action)
                done, reward, next_state = self.data_train.step(action.item())

                reward = torch.tensor(
                    [reward], dtype=torch.float, device=device)

                if next_state is not None:
                    next_state = torch.tensor(
                        [next_state], dtype=torch.float, device=device)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                if not done:
                    current_state = self.data_train.get_current_state()
                    state = torch.tensor(
                        [current_state], dtype=torch.float, device=device)

                # Perform one step of the optimization (on the target network)
                if self.windowed:
                    if self.is_strong_trend(current_state):
                        self.optimize_model_for_trend()
                    else:
                        self.optimize_model()
                else:
                    self.optimize_model()

                if done:
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        self.save_model(self.policy_net.state_dict())

        print('Complete')

    def save_model(self, model):
        torch.save(model, self.model_dir)

    def test(self, initial_investment=1000, test_type='test'):
        """
        :@param file_name: name of the .pkl file to load the model
        :@param test_type: test results on train data or test data
        :@return returns an Evaluation object to have access to different evaluation metrics.
        """
        data = self.data_train if test_type == 'train' else self.data_test

        self.test_net.load_state_dict(torch.load(self.model_dir))
        # print('opening:', self.model_dir)
        self.test_net.to(device)

        action_list = []
        data.__iter__()

        for batch in data:
            try:
                batch = batch.to(device)
                action_batch = self.test_net(batch).max(1)[1]
                action_list += list(action_batch.numpy())
                print(action_batch)
            except ValueError:
                action_list += [0]

        data.make_investment(action_list)
        ev_agent = Evaluation(data.data, data.action_name,
                              initial_investment, self.transaction_cost)
        print(test_type)
        ev_agent.evaluate()
        return ev_agent
