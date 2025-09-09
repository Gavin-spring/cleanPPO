# In src/solvers/ml/dnn_model.py
import torch
import torch.nn as nn

class KnapsackDNN(nn.Module):
    """
    Defines the neural network architecture for predicting knapsack solutions.
    """
    def __init__(self, input_size: int, config: dict = None):
        super(KnapsackDNN, self).__init__()
        dropout_rate = 0.5
        # Deeper Architecture
        self.layer1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.layer2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.layer3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.layer4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.layer5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.output_layer = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate) 

    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.layer3(x)))
        x = self.dropout(x)
        x = self.relu(self.bn4(self.layer4(x)))
        x = self.dropout(x)
        x = self.relu(self.bn5(self.layer5(x)))
        x = self.dropout(x)
        x = self.output_layer(x)
        return x