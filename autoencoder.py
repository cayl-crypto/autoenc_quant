import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(3, 3), bias=False)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0), bias=False)
        self.batch_norm1 = nn.BatchNorm2d(num_features=128)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0), bias=False)
        self.batch_norm2 = nn.BatchNorm2d(num_features=128)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False)
        self.batch_norm3 = nn.BatchNorm2d(num_features=256)
        self.leaky_relu4 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False)
        self.batch_norm4 = nn.BatchNorm2d(num_features=256)
        self.leaky_relu5 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False)
        self.batch_norm5 = nn.BatchNorm2d(num_features=512)
        self.leaky_relu6 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0), bias=False)
        self.batch_norm6 = nn.BatchNorm2d(num_features=512)
        self.leaky_relu7 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.flatten = nn.Flatten()


        # Decoder
        self.dense = nn.Linear(in_features=512, out_features=512*3*3)
        self.tconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0), bias=False)
        self.batch_norm7 = nn.BatchNorm2d(num_features=512)
        self.relu1 = nn.ReLU(inplace=True)
        self.tconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.batch_norm8 = nn.BatchNorm2d(num_features=256)
        self.relu2 = nn.ReLU(inplace=True)
        self.tconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.batch_norm9 = nn.BatchNorm2d(num_features=256)
        self.relu3 = nn.ReLU(inplace=True)
        self.tconv4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.batch_norm10 = nn.BatchNorm2d(num_features=128)
        self.relu4 = nn.ReLU(inplace=True)
        self.tconv5 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.batch_norm11 = nn.BatchNorm2d(num_features=64)
        self.relu5 = nn.ReLU(inplace=True)
        self.tconv6 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encode

        x = self.encode(x)  # x -> (Batch size, 512)

        # Decode
        x = self.decode(x) # x -> (Batch size, 3, 224, 224)

        return x

    def encode(self, x):
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        x = self.conv2(x)
        x = self.batch_norm1(x)
        x = self.leaky_relu2(x)
        x = self.conv3(x)
        x = self.batch_norm2(x)
        x = self.leaky_relu3(x)
        x = self.conv4(x)
        x = self.batch_norm3(x)
        x = self.leaky_relu4(x)
        x = self.conv5(x)
        x = self.batch_norm4(x)
        x = self.leaky_relu5(x)
        x = self.conv6(x)
        x = self.batch_norm5(x)
        x = self.leaky_relu6(x)
        x = self.conv7(x)
        x = self.leaky_relu7(x)
        x = x[:, :, 0, 0]
        return x

    def decode(self, x):
        x = self.dense(x)
        x = x.view(-1, 512, 3, 3)
        x = self.tconv1(x)
        x = self.batch_norm7(x)
        x = self.relu1(x)
        x = self.tconv2(x)
        x = self.batch_norm8(x)
        x = self.relu2(x)
        x = self.tconv3(x)
        x = self.batch_norm9(x)
        x = self.relu3(x)
        x = self.tconv4(x)
        x = self.batch_norm10(x)
        x = self.relu4(x)
        x = self.tconv5(x)
        x = self.batch_norm11(x)
        x = self.relu5(x)
        x = self.tconv6(x)
        x = self.tanh(x)
        return x

    def extract_feature(self, x):
        x = self.encode(x)
        return x
