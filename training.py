import glob
from tqdm import tqdm
import os
import csv
import torch
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(12321)

torch.cuda.manual_seed(12321)
torch.cuda.manual_seed_all(12321)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from utils import *

# Get dataset and labels.
seg_train = glob.glob("original/seg_train/seg_train/*/*.jpg")
seg_test = glob.glob("original/seg_test/seg_test/*/*.jpg")
classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
roots = ["original", "2", "4", "8", "16", "32", "64", "128", "256"]


def write_labels2csv(im_list, classes):
    labeli = []
    for im_path in tqdm(im_list):
        # read image

        # save path

        save_path_list = im_path.split(os.path.sep)
        c = save_path_list[-2]
        i = classes.index(c)
        save_path = os.path.join(*save_path_list[1:])
        label = [save_path, i]
        labeli.append(label)

    f = open('quant_dataset_labels_test.csv', 'w')

    with f:
        writer = csv.writer(f)
        writer.writerows(labeli)
    return labeli


# rlabels = write_labels2csv(seg_train, classes)
# rlabels = write_labels2csv(seg_test, classes)
# print(type(classes.index("buildings")))

def read_csv2labels(path):
    labeli = []
    f = open(path, 'r')

    with f:
        reader = csv.reader(f)
        for row in reader:
            labeli.append(row)
    return labeli


data_folder = "Dataset/"
labels_train = read_csv2labels(data_folder + 'shuffled_quant_dataset_labels_train.csv')
labels_test = read_csv2labels(data_folder + 'quant_dataset_labels_test.csv')
# for a in labels:
#     print(a[0], a[1])
# print()

# MODEL
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 34 * 34, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 34 * 34)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

import torch.optim as optim

# SHUFFLE DATASET
from sklearn.utils import shuffle

# labels_train = shuffle(labels_train)
# f = open('shuffled_quant_dataset_labels_train.csv', 'w')
#
# with f:
#     writer = csv.writer(f)
#     writer.writerows(labels_train)
# labels_test = shuffle(labels_test)

# TRAIN

import time as t


def train(net, root):
    result_list = [["Dataset", "Epoch", "Accuracy", "Time", "Total Correct", "Total Test Sample"]]
    print(root)
    net.to(device)
    net.cuda()
    batch_size = 64
    image_batch = torch.zeros(batch_size, 3, 150, 150)
    label_batch = torch.zeros(batch_size)
    label_batch = label_batch.type(torch.LongTensor)
    image_batch = image_batch.cuda()
    label_batch = label_batch.cuda()
    image_batch.to(device)
    label_batch.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    preprocess = transforms.Compose([
        transforms.Resize(150),
        transforms.CenterCrop(150),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    for epoch in tqdm(range(100)):
        start_time = t.time()
        # Extract all features with batches
        net.train()
        running_loss = 0.0
        for ids, data in enumerate(labels_train):
            # completes in about .
            im_path = data_folder + root + '/' + data[0]
            label = data[1]
            im = load_image(im_path)
            im = gray_to_rgb(im)
            im = preprocess(im)
            batch_index = ids % batch_size
            image_batch[batch_index] = im
            label_batch[batch_index] = int(label)

            if ids == len(labels_train) - 1:
                image_batch_last = image_batch[:batch_index + 1]
                label_batch_last = label_batch[:batch_index + 1]
                optimizer.zero_grad()
                outputs = net(image_batch_last)
                loss = criterion(outputs, label_batch_last)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                print('loss: %f' %
                      (running_loss / 14034))
                running_loss = 0.0
                # break

            if batch_index == batch_size - 1:
                optimizer.zero_grad()
                outputs = net(image_batch)
                loss = criterion(outputs, label_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

        net.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for ids, data in enumerate(labels_test):
                # completes in about .
                im_path = data_folder + root + '/' + data[0]
                label = data[1]
                im = load_image(im_path)
                im = gray_to_rgb(im)
                im = preprocess(im)
                batch_index = ids % batch_size
                image_batch[batch_index] = im
                label_batch[batch_index] = int(label)

                if ids == len(labels_test) - 1:
                    image_batch_last = image_batch[:batch_index + 1]
                    label_batch_last = label_batch[:batch_index + 1]
                    outputs = net(image_batch_last)
                    _, predicted = torch.max(outputs, 1)
                    total += label_batch_last.size(0)
                    res = sum(predicted == label_batch_last)
                    correct += res.item()
                    break

                if batch_index == batch_size - 1:
                    outputs = net(image_batch)
                    _, predicted = torch.max(outputs, 1)
                    total += label_batch.size(0)
                    res = sum(predicted == label_batch)
                    correct += res.item()

            result_list.append([root, epoch + 1, correct / total, t.time() - start_time, correct, total])

    f = open(root + '_results.csv', 'w')

    with f:
        writer = csv.writer(f)
        writer.writerows(result_list)

    return 0


total_time = t.time()

# Train Original
# original_net = Net()
# result_list = train(original_net, roots[0])
# # Train 2
# _2_net = Net()
# result_list = train(_2_net, roots[1])
# # # Train 4
# _4_net = Net()
# result_list = train(_4_net, roots[2])
# # # Train 8
# _8_net = Net()
# result_list = train(_8_net, roots[3])
# # # Train 16
# _16_net = Net()
# result_list = train(_16_net, roots[4])
# # # Train 32
# _32_net = Net()
# result_list = train(_32_net, roots[5])
# # # Train 64
# _64_net = Net()
# result_list = train(_64_net, roots[6])
# # # Train 128
# _128_net = Net()
# result_list = train(_128_net, roots[7])
# # # Train 256
# _256_net = Net()
# result_list = train(_256_net, roots[8])
print(t.time() - total_time)
print('Finished Training')
