"""
==========================================
Copyright (C) 2023 Pattern Recognition and Machine Learning Group   
All rights reserved
Description:
Created by Li Wei at 2023/7/27 10:57
Email:1280358009@qq.com
==========================================
"""
import torch
import torch.nn as nn


class DilatedConvNet(nn.Module):
    def __int__(self, input_channels, output_channels):
        super(DilatedConvNet, self).__init__()

        self.dconv1 = nn.conv2d(input_channels, output_channels, kernel_size=3, dilation=1)
        self.dconv2 = nn.conv2d(input_channels, output_channels, kernel_size=3, dilation=2)
        self.dconv3 = nn.conv2d(input_channels, output_channels, kernel_size=3, dilation=4)

        self.weights = nn.Parameter(torch.Tensor(3))

        nn.init.normal(self.weights)

    def forward(self, x):
        out1 = self.dconv1(x)
        out2 = self.dconv2(x)
        out3 = self.dconv3(x)

        merged_spatial_output = self.weights[0] * out1 + self.weights[1] * out2 + self.weights[2] * out3

        return merged_spatial_output


class LSTMWithAttention(nn.Module):
    def __int__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMWithAttention, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            # Initialize LSTM hidden and cell states
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

            # Forward propagate LSTM
            lstm_out, _ = self.lstm(x, (h0, c0))

            # Apply attention mechanism
            attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
            attention_out = torch.sum(attention_weights * lstm_out, dim=1)

            # Fully connected layer
            output = self.fc(attention_out)

            return output


class MVA_DCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MVA_DCN, self).__init__()

        self.window_length = input_size[1]  # Number of timestamps in the input
        self.dilated_conv1 = DilatedConvNet(input_size[2:], ...)  # Create an instance of the DilatedConvModel
        self.pooling = nn.MaxPool2d(kernel_size=(2, 2))
        self.dilated_conv2 = DilatedConvNet(input_size[2:], ...)  # Create an instance of the DilatedConvModel
        self.lstm_attention = LSTMWithAttention(..., hidden_size, num_layers,
                                                output_size)  # Create an instance of the LSTMWithAttention

    def forward(self, x):
        spatial_features = []

        # Iterate over the window length
        for i in range(self.window_length):
            # Extract the spatial feature at each timestamp using the DilatedConvModel
            spatial_feature = self.dilated_conv1(x[:, i, ...])

            # Apply pooling to keep the size unchanged
            spatial_feature = self.pooling(spatial_feature)

            # Pass the spatial feature through the second DilatedConvModel
            spatial_feature = self.dilated_conv2(spatial_feature)

            # Add the input to the second DilatedConvModel as a residual connection
            spatial_feature = spatial_feature + x[:, i, ...]

            spatial_features.append(spatial_feature.unsqueeze(1))

        # Concatenate the spatial features along the temporal dimension
        spatial_features = torch.cat(spatial_features, dim=1)

        # Pass the spatial features through the LSTM with Attention module
        output = self.lstm_attention(spatial_features)

        return output


# # Set hyperparameters
# input_size = (32, 12, 2, 32, 32)  # (batch, window_length, channels, length, width)
# hidden_size = 64
# num_layers = 2
# output_size = 1
#
# # Create an instance of the modified combined model
# model = MVA_DCN(input_size, hidden_size, num_layers, output_size)
#
# # Create a random input tensor
# batch_size = input_size[0]
# input_tensor = torch.randn(input_size)
#
# # Forward pass
# output = model(input_tensor)
#
# # Print the output shape
# print("Output shape:", output.shape)