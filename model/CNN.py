import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    input shape - batch, channel, seq_len
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CNN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim,
                      out_channels=hidden_dim[0],
                      padding=1,
                      stride=1,
                      kernel_size=4),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim[0],
                      out_channels=hidden_dim[1],
                      padding=1,
                      stride=1,
                      kernel_size=4),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim[1],
                      out_channels=hidden_dim[2],
                      padding=1,
                      stride=1,
                      kernel_size=4),
            nn.ReLU()
        )
        self.conv1 = nn.Conv1d(in_channels=hidden_dim[2],
                               out_channels=output_dim,
                               padding=0,
                               stride=1,
                               kernel_size=1)

    def forward(self, inp):
        output = self.block1(inp)
        output = self.block2(output)
        output = self.block3(output)
        output = F.adaptive_avg_pool1d(output, (1))
        output = self.conv1(output)
        return output