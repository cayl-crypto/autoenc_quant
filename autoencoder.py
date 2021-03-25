import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


#
# # Converting data to torch.FloatTensor
# transform = transforms.ToTensor()
#
# # Download the training and test datasets
# train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
#
# test_data = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
#
# # Prepare data loaders
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, num_workers=0)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, num_workers=0)
#
#
# # Utility functions to un-normalize and display an image
# def imshow(img):
#     img = img / 2 + 0.5
#     plt.imshow(np.transpose(img, (1, 2, 0)))
#
#
# # Define the image classes
# classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#
# # Obtain one batch of training images
# dataiter = iter(train_loader)
# images, labels = dataiter.next()
# images = images.numpy()  # convert images to numpy for display
#
# # Plot the images
# fig = plt.figure(figsize=(8, 8))
# # display 20 images
# for idx in np.arange(9):
#     ax = fig.add_subplot(3, 3, idx + 1, xticks=[], yticks=[])
#     imshow(images[idx])
#     ax.set_title(classes[labels[idx]])
#

# Define the Convolutional Autoencoder
# class ConvAutoencoder(nn.Module):
#     def __init__(self):
#         super(ConvAutoencoder, self).__init__()
#
#         # Encoder
#         self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
#         self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
#         self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.output_pool = nn.MaxPool2d(40, 40)
#
#         # Decoder
#         self.t_conv1 = nn.ConvTranspose2d(256, 128, 3, stride=2)
#         self.t_conv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
#         self.t_conv3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
#         self.t_conv4 = nn.ConvTranspose2d(32, 16, 2, stride=1)
#         self.t_conv5 = nn.ConvTranspose2d(16, 3, 2, stride=1, padding=3)
#         self.flat = nn.Flatten()
#         # self.up = nn.Upsample(scale_factor=2, mode='nearest')
#
#     def forward(self, x):
#         input_size = x.size()
#         # print(input_size)
#         x = F.relu(self.conv1(x))
#         # print(x.size())
#         # x = self.pool(x)
#         # print(x.size())
#         x = F.relu(self.conv2(x))
#         # print(x.size())
#         x = self.pool(x)
#         # print(x.size())
#         x = F.relu(self.conv3(x))
#         # print(x.size())
#         x = self.pool(x)
#         # print(x.size())
#         x = F.relu(self.conv4(x))
#         # print(x.size())
#         # x = self.pool(x)
#         # print(x.size())
#         x = F.relu(self.conv5(x))
#         # print(x.size())
#         x = self.pool(x)
#
#         x = F.relu(self.t_conv1(x))
#         x = F.relu(self.t_conv2(x))
#         x = F.relu(self.t_conv3(x))
#         x = F.relu(self.t_conv4(x))
#         # print(x.size())
#         x = torch.sigmoid(self.t_conv5(x, output_size=input_size))
#         # print(x.size())
#
#         return x
#
#     def encode(self, x):
#         x = F.relu(self.conv1(x))
#         # print(x.size())
#         # x = self.pool(x)
#         # print(x.size())
#         x = F.relu(self.conv2(x))
#         # print(x.size())
#         x = self.pool(x)
#         # print(x.size())
#         x = F.relu(self.conv3(x))
#         # print(x.size())
#         x = self.pool(x)
#         # print(x.size())
#         x = F.relu(self.conv4(x))
#         # print(x.size())
#         # x = self.pool(x)
#         # print(x.size())
#         x = F.relu(self.conv5(x))
#         x = self.output_pool(x)
#         x = self.flat(x)
#
#         return x

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.batch_norm1 = nn.BatchNorm2d(num_features=128)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.batch_norm2 = nn.BatchNorm2d(num_features=256)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.batch_norm3 = nn.BatchNorm2d(num_features=512)
        self.leaky_relu4 = nn.LeakyReLU(negative_slope=0.2, inplace=True)


        # Decoder
        self.tconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0), bias=False)
        self.batch_norm4 = nn.BatchNorm2d(num_features=512)
        self.relu1 = nn.ReLU(inplace=True)
        self.tconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.batch_norm5 = nn.BatchNorm2d(num_features=256)
        self.relu2 = nn.ReLU(inplace=True)
        self.tconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.batch_norm6 = nn.BatchNorm2d(num_features=128)
        self.relu3 = nn.ReLU(inplace=True)
        self.tconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.batch_norm7 = nn.BatchNorm2d(num_features=64)
        self.relu4 = nn.ReLU(inplace=True)
        self.tconv5 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.tanh = nn.Tanh()

        self.avg_pool = nn.AvgPool2d(kernel_size=4)

    def forward(self, x):
        # input = (Batch size, 3, 160, 160)
        # Encode

        x = self.encode(x)  # x -> (Batch size, 512)

        # Add width and height
        # x = torch.unsqueeze(torch.unsqueeze(x, -1), -1)  # x -> (Batch size, 512, 1, 1)
        x = self.avg_pool(x)
        # x = x[:, :, 0, 0]
        # Decode
        x = self.decode(x) # x -> (Batch size, 3, 160, 160)

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
        return x

    def decode(self, x):
        x = self.tconv1(x)
        x = self.batch_norm4(x)
        x = self.relu1(x)
        x = self.tconv2(x)
        x = self.batch_norm5(x)
        x = self.relu2(x)
        x = self.tconv3(x)
        x = self.batch_norm6(x)
        x = self.relu3(x)
        x = self.tconv4(x)
        x = self.batch_norm7(x)
        x = self.relu4(x)
        x = self.tconv5(x)
        x = self.tanh(x)
        return x

    def extract_feature(self, x):
        x = self.encode(x)
        x = self.avg_pool(x)
        return x[:, :, 0, 0]
