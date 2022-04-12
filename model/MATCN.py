import torch
import torch.nn as nn
import torch.nn.functional as F


class MATCNModel(nn.Module):
    def __init__(self, tcn_layer_num, tcn_kernel_size, tcn_input_dim, tcn_filter_num, window_size, forecast_horizon, num_ouput_time_series,
                 use_bias, tcn_dropout_rate):
        super(MATCNModel, self).__init__()

        self.num_outpput_time_sereis = num_ouput_time_series

        self.lower_tcn = TCNStack(tcn_layer_num, tcn_input_dim, tcn_filter_num, tcn_kernel_size, use_bias,tcn_dropout_rate)
        self.downsample_att = DownsampleLayerWithAttention(num_ouput_time_series, tcn_filter_num, window_size, tcn_kernel_size, forecast_horizon)

    def forward(self, input_tensor):
        x = self.lower_tcn(input_tensor)
        x, distribution = self.downsample_att([x, input_tensor])
        return x


class SepDenseLayer(nn.Module):
    def __init__(self, num_input_dim, window_size, output_size, use_bias):
        super(SepDenseLayer, self).__init__()
        self.activation = nn.ReLU()
        self.use_bias = use_bias

        self.w = nn.Parameter(torch.normal(mean=0.0, std=0.05, size=(num_input_dim, output_size, window_size)))
        if self.use_bias:
            self.b = nn.Parameter(torch.zeros(size=(num_input_dim, output_size)))

    def forward(self, x):
        output = torch.matmul(x, self.w)
        output = torch.squeeze(output, 2)
        if self.use_bias:
            output = torch.add(output, self.b)
        return self.activation(output)


class DownsampleLayerWithAttention(nn.Module):
    def __init__(self, num_output_time_series, filter_num, window_size, kernel_size, output_size):
        super(DownsampleLayerWithAttention, self).__init__()

        self.down_tcn = nn.Conv1d(in_channels=filter_num,
                                  out_channels=num_output_time_series,
                                  kernel_size=kernel_size)
        self.weight_norm_down_tcn = nn.utils.weight_norm(self.down_tcn)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.query_dense_layer = nn.Linear(num_output_time_series, output_size)
        self.key_dense_layer = SepDenseLayer(num_output_time_series, window_size, window_size, use_bias=True)
        self.value_dense_layer = SepDenseLayer(num_output_time_series, window_size, window_size, use_bias=False)
        self.post_attention_layer = DotAttentionLayer(window_size)

    def forward(self, input_tensors):
        tcn_out = self.weight_norm_down_tcn(input_tensors[0])
        tcn_out = self.gap(tcn_out)
        tcn_out = tcn_out.permute(0, 2, 1).contiguous()

        or_input = input_tensors[1]

        query = self.query_dense_layer(tcn_out)
        key = self.key_dense_layer(torch.unsqueeze(or_input, -2))
        value = self.value_dense_layer(torch.unsqueeze(or_input, -2))

        x, distribution = self.post_attention_layer([query, value, key])
        return x, distribution


class DotAttentionLayer(nn.Module):
    def __init__(self, scale_value):
        super(DotAttentionLayer, self).__init__()
        self.scale_value = torch.tensor(scale_value, dtype=torch.float32)

    def forward(self, tensors):
        value = tensors[1]
        key = tensors[2].permute(0, 2, 1).contiguous()
        query = tensors[0]
        scores = torch.matmul(query, key)
        scores = scores / torch.sqrt(self.scale_value)
        distribution = F.softmax(scores, dim=1)
        output = torch.squeeze(torch.matmul(distribution, value))
        return output, distribution


class BasicTCNBlock(nn.Module):
    def __init__(self, input_dim, filter_num, kernel_size, dilation_rate, use_bias, dropout_rate):
        super(BasicTCNBlock, self).__init__()

        self.tcn_1 = nn.Conv1d(input_dim, filter_num, kernel_size=kernel_size, dilation=dilation_rate, bias=use_bias)
        self.weight_norm_layer1 = nn.utils.weight_norm(self.tcn_1)

        self.tcn_2 = nn.Conv1d(filter_num, filter_num, kernel_size=kernel_size, dilation=dilation_rate, bias=use_bias)
        self.weight_norm_layer2 = nn.utils.weight_norm(self.tcn_2)

        self.tcn_3 = nn.Conv1d(input_dim, filter_num, kernel_size=1, dilation=dilation_rate, bias=use_bias)
        self.weight_norm_layer3 = nn.utils.weight_norm(self.tcn_3)

        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.activation = nn.ReLU()

    def forward(self, input_tensor):
        x = self.weight_norm_layer1(input_tensor)
        x = self.activation(x)
        x = self.dropout_layer(x)
        x = self.weight_norm_layer2(x)
        x = self.activation(x)
        x = self.dropout_layer(x)
        res = self.weight_norm_layer3(input_tensor)
        res = nn.AdaptiveAvgPool1d(x.shape[-1])(res)
        x = torch.add(res, x)
        return x


class TCNStack(nn.Module):
    def __init__(self, layer_num, input_dim, filter_num, kernel_size, use_bias, dropout_rate):
        super(TCNStack, self).__init__()

        self.block_seq = [BasicTCNBlock(input_dim, filter_num, kernel_size, 1, use_bias, dropout_rate)]
        for i in range(1, layer_num - 1):
            self.block_seq.append(BasicTCNBlock(filter_num, filter_num, kernel_size, 2 ** i, use_bias, dropout_rate))
        self.block_seq.append(BasicTCNBlock(filter_num, filter_num, kernel_size, 2 ** (layer_num - 1), use_bias, dropout_rate))
        self.block_seq = nn.Sequential(*self.block_seq)

    def forward(self, input_tensor):
        x = self.block_seq(input_tensor)
        return x

if __name__ == "__main__":
    x = torch.randn(256, 3, 32)
    model = MATCNModel(3, 3, 3, 64, 32, 32, 3, True, 0.3)
    x, distribution = model(x)
    print(x)
    # for i in range(3):
