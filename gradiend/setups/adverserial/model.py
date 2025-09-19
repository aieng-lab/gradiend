import torch
import torch.nn as nn
import torch.optim as optim


class SimpleBinaryClassifier(nn.Module):
    def __init__(self, in_features=4):
        super().__init__()
        # single linear projection to 1 logit
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        # x: batch of shape (batch_size, 4)
        logits = self.fc(x)  # shape (batch_size, 1)
        return logits

class OneHotBinaryClassifier(nn.Module):
    def __init__(self, in_features=4):
        super().__init__()
        self.fc = nn.Linear(in_features, 2)  # 2 outputs for one-hot

    def forward(self, x):
        return self.fc(x)  # logits, no softmax

class TwoOutputRegressor(nn.Module):
    def __init__(self, in_features=4):
        super().__init__()
        self.fc = nn.Linear(in_features, 2)  # two outputs

    def forward(self, x): # todo probably use tanh or other activation functions to evaluate their influence
        return self.fc(x)  # shape (B, 2), raw continuous values


if __name__ == '__main__':
    # Usage example
    model = SimpleBinaryClassifier(in_features=4)
    criterion = nn.BCEWithLogitsLoss()  # binary cross‐entropy on probabilities
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # in a training step:
    #   inputs: tensor(shape=(B,4))
    #   targets: tensor(shape=(B,1)) with 0/1 labels
    # logits = model(inputs)
    # loss   = criterion(logits, targets)
    # loss.backward(); optimizer.step()
