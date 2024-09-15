import torch.optim as optim
from src.agents.mlp_encoder.seq2seq import Seq2Seq
from src.agents.mlp_encoder.decoder import Decoder
from src.agents.mlp_encoder.encoder import Encoder
from src.agents.base_train import BaseTrain
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Train(BaseTrain):
    def __init__(self,
                 data_loader,
                 data_train,
                 data_test,
                 dataset_name,
                 window_size=1,
                 transaction_cost=0.0,
                 n_classes=50,
                 BATCH_SIZE=30,
                 GAMMA=0.7,
                 ReplayMemorySize=50,
                 TARGET_UPDATE=5,
                 n_step=10):

        super(Train, self).__init__(data_loader,
                                    data_train,
                                    data_test,
                                    dataset_name,
                                    'simpleMLP',
                                    window_size,
                                    transaction_cost,
                                    BATCH_SIZE,
                                    GAMMA,
                                    ReplayMemorySize,
                                    TARGET_UPDATE,
                                    n_step,
                                    windowed=True)

        self.encoder = Encoder(n_classes, data_train.state_size).to(device)
        self.policy_decoder = Decoder(n_classes, 3).to(device)
        self.target_decoder = Decoder(n_classes, 3).to(device)

        self.policy_net = Seq2Seq(self.encoder, self.policy_decoder).to(device)
        self.target_net = Seq2Seq(self.encoder, self.target_decoder).to(device)

        self.target_decoder.load_state_dict(self.policy_decoder.state_dict())
        self.target_decoder.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)

        test_encoder = Encoder(
            n_classes, self.data_train.state_size).to(device)
        test_decoder = Decoder(n_classes, 3).to(device)

        self.test_net = Seq2Seq(test_encoder, test_decoder)
        self.test_net.to(device)
