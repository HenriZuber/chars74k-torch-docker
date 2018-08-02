import torch  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import numpy as np

# Classes et fonctions utilisees dans les modeles


def flatten(x):
    N = x.shape[0]  # read in N, C, H, W
    return x.view(
        N, -1
    )  # "flatten" the C * H * W values into a single vector per image


def weights_init_conv(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal(module.weight, nonlinearity="relu")
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal(module.weight)
    elif isinstance(module, Linear_then_softmax):
        pass
    elif isinstance(module, Linear_then_no_softmax):
        pass


class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)


class Linear_then_softmax(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        nn.init.xavier_normal(self.lin.weight)
        self.soft = nn.LogSoftmax()

    def forward(self, x):
        out = self.lin(x)
        out = self.soft(out)
        return out


class Linear_then_no_softmax(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        nn.init.xavier_normal(self.lin.weight)

    def forward(self, x):
        out = self.lin(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=5,
            stride=stride,
            padding=2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# Models


class simple_model(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=5,
            stride=stride,
            padding=2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.fc1 = nn.Linear(64 * 64 * out_channels, out_channels)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = flatten(out)
        out = self.fc1(out)

        return out


class medium_model(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, 128, kernel_size=5, stride=stride, padding=2, bias=False
        )
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            128, 128, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(
            128, 128, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(
            128, 128, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(128)

        self.mp = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(
            128, 64, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(
            64, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn6 = nn.BatchNorm2d(out_channels)

        self.fc1 = nn.Linear(15872, 2048)

        self.do = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(2048, out_channels)

    def forward(self, x):

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        out = self.mp(out)

        out = self.conv4(out)
        out = self.relu(out)
        out = self.bn4(out)

        out = self.conv5(out)
        out = self.relu(out)
        out = self.bn5(out)
        out = self.mp(out)

        out = self.conv6(out)
        out = self.relu(out)
        out = self.bn6(out)

        out = flatten(out)
        out = self.do(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.do(out)

        out = self.fc2(out)
        out = self.relu(out)

        return out


def res_model(in_channels, out_channels):
    model = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        ResidualBlock(64, 64),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=5, padding=2),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.MaxPool2d(2),
        Flatten(),
        nn.Linear(32768, 2048),
        nn.ReLU(),
        nn.Dropout(p=0.8),
        nn.Linear(2048, out_channels),
    )
    return model


def advanced_model(in_channels, out_channels, no_softmax=True):
    model = nn.Sequential(
        nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(128),  # ajout
        nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=2),  # 128,256
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=2),  # ajout
        nn.ReLU(),  # ajout
        nn.MaxPool2d(2),
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 256,256
        nn.ReLU(),
        nn.BatchNorm2d(256),  # ajout
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        Flatten(),
        nn.Linear(512, 2048),
        nn.ReLU(),
        nn.Dropout(p=0.8),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Dropout(p=0.8)
    )

    if no_softmax:
        model = nn.Sequential(model, Linear_then_no_softmax(2048, 62))
    else:
        model = nn.Sequential(model, Linear_then_softmax(2048, 62))

    return model


def mid_model(in_channels, out_channels):
    model = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=5, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=5, stride=2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        Flatten(),
        nn.Linear(4608, 2048),  # 4608,2048
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, out_channels),
    )
    return model
