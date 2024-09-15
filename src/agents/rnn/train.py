import torch
import torch.optim as optim
from src.agents.rnn.rnn import RNN_DQN
from src.agents.base_train import BaseTrain

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Train(BaseTrain):
    def __init__(self,
                 data_loader,
                 data_train,
                 data_test,
                 dataset_name,
                 window_size=1,
                 transaction_cost=0.0,
                 BATCH_SIZE=30,
                 GAMMA=0.7,
                 ReplayMemorySize=50,
                 TARGET_UPDATE=5,
                 n_step=10,
                 num_layers=1):

        super(Train, self).__init__(data_loader,
                                    data_train,
                                    data_test,
                                    dataset_name,
                                    'RNN_DQN',
                                    window_size,
                                    transaction_cost,
                                    BATCH_SIZE,
                                    GAMMA,
                                    ReplayMemorySize,
                                    TARGET_UPDATE,
                                    n_step)

        self.policy_net = RNN_DQN(
            data_train.state_size, 3, num_layers).to(device)
        self.target_net = RNN_DQN(
            data_train.state_size, 3, num_layers).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)

        self.test_net = RNN_DQN(
            self.data_train.state_size, 3, num_layers).to(device)
