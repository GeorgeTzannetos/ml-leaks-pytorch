import torch.nn as nn
import torch.nn.functional as F
import torch


# Define the ConvNet model used for building the target model. The same structure is used for the shadow model as well
# based on the first attack of the paper's section III.


class ConvNet(nn.Module):
    """ Model has two convolutional layers, two pooling  layers and a hidden layer with 128 units."""
    def __init__(self, input_size=3):
        super(ConvNet, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=48, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(3, 3))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        if input_size == 3:
            self.fc_features = 6*6*96
        else:
            self.fc_features = 5*5*96
        self.fc1 = nn.Linear(in_features=self.fc_features, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, self.fc_features)  # reshape x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MlleaksMLP(torch.nn.Module):
    """
    This is a simple multilayer perceptron with 64-unit hidden layer and a softmax output layer.
    """
    def __init__(self, input_size=3, hidden_size=64, output=1):
        super(MlleaksMLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        output = self.fc2(hidden)
        output = self.sigmoid(output)
        return output
