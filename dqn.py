import torch
from torch import nn

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class DQN(nn.Module):
    """
    Klassisches Dueling Q-Network optimiert für wertbasierte RL-Stabilität.
    """
    def __init__(self, input_size=6, output_size=3, hidden_size=512, value_hidden_size=None, adv_hidden_size=None):
        super().__init__()
        
        if value_hidden_size is None:
            value_hidden_size = hidden_size // 2
        if adv_hidden_size is None:
            adv_hidden_size = hidden_size // 2
        
        # Lineare Feature-Extraktion ohne Layer-Norm für schnellere Wertkonvergenz
        self.feature = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # Value Stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, value_hidden_size),
            nn.ReLU(),
            nn.Linear(value_hidden_size, 1),
        )
        
        # Advantage Stream
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden_size, adv_hidden_size),
            nn.ReLU(),
            nn.Linear(adv_hidden_size, output_size),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        adv = self.adv_stream(features)
        return value + (adv - adv.mean(dim=1, keepdim=True))