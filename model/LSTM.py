import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, gpu=True):
        super(LSTM, self).__init__()
        """
        input shape - batch, seq_len, channel
        """
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gpu = gpu

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, inp):
        h0, c0 = self.init_hidden(inp)

        output, (h_t, c_t) = self.lstm(inp, (h0, c0))
        output = self.bn(output[:, -1, :])
        output = torch.flatten(output, start_dim=1)
        output = self.fc(output)
        return output

    def init_hidden(self, inp):
        b, seq_len, c = inp.size()
        if self.num_layers == 2:
            h0 = torch.zeros(2, b, self.hidden_dim)
            c0 = torch.zeros(2, b, self.hidden_dim)
        else:
            h0 = torch.zeros(1, b, self.hidden_dim)
            c0 = torch.zeros(1, b, self.hidden_dim)
        if self.gpu:
            return [t.cuda() for t in (h0, c0)]
        else:
            return [t for t in (h0, c0)]