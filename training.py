import glob
from tqdm import tqdm
import os
import csv
import torch
import torchvision
import torchvision.transforms as transforms
from autoencoder import ConvAutoencoder

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


# for a in labels:
#     print(a[0], a[1])
# print()

# MODEL
import torch.nn as nn

data_folder = "Dataset/"
labels_train = read_csv2labels(data_folder + 'shuffled_quant_dataset_labels_train.csv')
labels_test = read_csv2labels(data_folder + 'shuffled_quant_dataset_labels_test.csv')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

import torch.optim as optim

# SHUFFLE DATASET
# from sklearn.utils import shuffle
#
# labels_train = shuffle(labels_train)
# f = open(data_folder + 'shuffled_quant_dataset_labels_train.csv', 'w')
#
# with f:
#     writer = csv.writer(f)
#     writer.writerows(labels_train)
# labels_test = shuffle(labels_test)
# f = open(data_folder + 'shuffled_quant_dataset_labels_test.csv', 'w')
#
# with f:
#     writer = csv.writer(f)
#     writer.writerows(labels_test)

# TRAIN

import time as t


def train(net, root):
    max_test_loss = 9999.9
    result_list = [["Dataset", "Epoch", "Time", "Total Train Loss", "Total Test Loss"]]
    print(root)
    net.to(device)
    net.cuda()
    batch_size = 64
    image_batch_i = torch.zeros(batch_size, 3, 150, 150)
    image_batch_i = image_batch_i.cuda()
    image_batch_i.to(device)

    image_batch_o = torch.zeros(batch_size, 3, 150, 150)
    image_batch_o = image_batch_o.cuda()
    image_batch_o.to(device)

    image_batch_t_i = torch.zeros(batch_size, 3, 150, 150)
    image_batch_t_i = image_batch_t_i.cuda()
    image_batch_t_i.to(device)

    image_batch_t_o = torch.zeros(batch_size, 3, 150, 150)
    image_batch_t_o = image_batch_t_o.cuda()
    image_batch_t_o.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    preprocess_i = transforms.Compose([
        transforms.Resize(150),
        transforms.CenterCrop(150),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    preprocess_o = transforms.Compose([
        transforms.Resize(150),
        transforms.CenterCrop(150),
        transforms.ToTensor(),
    ])

    for epoch in tqdm(range(1)):
        # i stands for input, o stands for output
        start_time = t.time()
        # Extract all features with batches
        net.train()
        total_train_loss = 0.0
        for ids, data in enumerate(labels_train):
            # completes in about .
            img_path = data_folder + root + '/' + data[0]
            img = load_image(img_path)
            img = gray_to_rgb(img)
            im_i = preprocess_i(img)
            im_o = preprocess_o(img)
            batch_index = ids % batch_size
            image_batch_i[batch_index] = im_i
            image_batch_o[batch_index] = im_o

            if ids == len(labels_train) - 1:
                image_batch_last_i = image_batch_i[:batch_index + 1]
                image_batch_last_o = image_batch_o[:batch_index + 1]
                optimizer.zero_grad()
                results = net(image_batch_last_i)
                loss = criterion(results, image_batch_last_o)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                print('loss: %f' %
                      (total_train_loss / 14034))

            if batch_index == batch_size - 1:
                optimizer.zero_grad()
                results = net(image_batch_i)
                # encoded features are x
                # x = net.encode(image_batch_i)
                ########################
                loss = criterion(results, image_batch_o)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

        net.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for ids, data in enumerate(labels_test):
                # completes in about .
                im_path = data_folder + root + '/' + data[0]
                label = data[1]
                im = load_image(im_path)
                im = gray_to_rgb(im)
                im_i = preprocess_i(im)
                im_o = preprocess_o(im)
                batch_index = ids % batch_size
                image_batch_t_i[batch_index] = im_i
                image_batch_t_o[batch_index] = im_o

                if ids == len(labels_test) - 1:
                    image_batch_last_t_i = image_batch_t_i[:batch_index + 1]
                    image_batch_last_t_o = image_batch_t_o[:batch_index + 1]
                    outputs = net(image_batch_last_t_i)
                    loss = criterion(outputs, image_batch_last_t_o)
                    total_test_loss += loss.item()
                    if total_test_loss < max_test_loss:
                        # save best model
                        torch.save(net, "best_" + root + ".pt")
                        max_test_loss = total_test_loss
                    break

                if batch_index == batch_size - 1:
                    outputs = net(image_batch_t_i)
                    loss = criterion(outputs, image_batch_t_o)
                    total_test_loss += loss.item()

            result_list.append(
                [root, epoch + 1, t.time() - start_time, str(total_train_loss / 14034), str(total_test_loss / 3000)])

    f = open(root + '_results.csv', 'w')

    with f:
        writer = csv.writer(f)
        writer.writerows(result_list)

    return 0


total_time = t.time()

# Train Original
original_net = ConvAutoencoder()
print(original_net)
original_net = train(original_net, roots[0])

# Train 2
_2_net = ConvAutoencoder()
result_list = train(_2_net, roots[1])
# # Train 4
_4_net = ConvAutoencoder()
result_list = train(_4_net, roots[2])
# # Train 8
_8_net = ConvAutoencoder()
result_list = train(_8_net, roots[3])
# # Train 16
_16_net = ConvAutoencoder()
result_list = train(_16_net, roots[4])
# # Train 32
_32_net = ConvAutoencoder()
result_list = train(_32_net, roots[5])
# # Train 64
_64_net = ConvAutoencoder()
result_list = train(_64_net, roots[6])
# # Train 128
_128_net = ConvAutoencoder()
result_list = train(_128_net, roots[7])
# # Train 256
_256_net = ConvAutoencoder()
result_list = train(_256_net, roots[8])
print(t.time() - total_time)
print('Finished Training')

# preprocess = transforms.Compose([
#     transforms.Resize(150),
#     transforms.CenterCrop(150),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# im_path = "20057.jpg"
# im = load_image(im_path)
# im = gray_to_rgb(im)
# im = preprocess(im)
# im = torch.unsqueeze(im, 0)
#
# im = im.cuda()
# im.to(device)
# original_net.eval()
# with torch.no_grad():
#     output = original_net(im)
#     output = torch.squeeze(output, 0)
#     s = transforms.ToPILImage()
#     # show_image(output.permute(1, 2, 0).numpy())
#     print(output.shape)
#     show_image(s(output))
