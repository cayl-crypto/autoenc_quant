# Train Autoencoders for mscoco dataset
import glob
from pathlib import Path
from tqdm import tqdm
import os
import csv
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from autoencoder import ConvAutoencoder
import time as t

torch.manual_seed(12321)

torch.cuda.manual_seed(12321)
torch.cuda.manual_seed_all(12321)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from utils import *

# Get dataset paths.
script_dir = Path().absolute()
project_dir = script_dir.parent
roots = ["Resized", "quant_learning/2", "quant_learning/4", "quant_learning/8"]
# print(project_dir.joinpath(roots[0]))

preprocess_input = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

preprocess_output = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


# print(nets)
def get_image_paths(dataset_path, train_images=True):
    # inputs: Dataset folder and inside folder for mscoco (train2017 or val2017).
    # Go parent folder and get datasets from there.
    if train_images:
        dataset_path = Path().absolute().parent.joinpath(dataset_path, "train2017", "*.jpg")
    else:
        dataset_path = Path().absolute().parent.joinpath(dataset_path, "val2017", "*.jpg")
    # get paths of images.

    return glob.glob(str(dataset_path))


def train_step(model, input_batch, output, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    results = model(input_batch)
    loss = criterion(results, output)
    loss.backward()
    optimizer.step()
    # print(loss.item())
    return loss.item()
    # raise NotImplementedError


def test(model, input_batch, output, criterion):
    model.eval()
    with torch.no_grad():
        results = model(input_batch)
    loss = criterion(results, output)
    return loss.item()
    # raise NotImplementedError


def train(model, dataset_path, epoch, batch_size, model_name):
    # model = nn.DataParallel(model)
    model.to(device)
    train_image_paths = get_image_paths(dataset_path=dataset_path)
    test_image_paths = get_image_paths(dataset_path=dataset_path, train_images=False)

    image_batch_input = torch.zeros(batch_size, 3, 224, 224).to(device)


    # criterion = nn.BCELoss()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    prev_test_loss = 99999.0

    for i in tqdm(range(epoch)):

        total_train_loss = 0.0
        for ids, train_image_path in enumerate(train_image_paths):
            img = load_image(train_image_path)
            img = gray_to_rgb(img)
            img_input = preprocess_input(img)
            img_output = preprocess_output(img)
            batch_index = ids % batch_size
            image_batch_input[batch_index] = img_input

            if batch_index == batch_size - 1:
                total_train_loss += train_step(model=model, input_batch=image_batch_input, output=image_batch_input,
                                               criterion=criterion, optimizer=optimizer) * batch_index

            if ids == len(train_image_paths) - 1:
                total_train_loss += train_step(model=model, input_batch=image_batch_input[:batch_index + 1],
                                               output=image_batch_input[:batch_index + 1],
                                               criterion=criterion, optimizer=optimizer) * batch_index
                print('train loss: %f' %
                      (total_train_loss / len(train_image_paths)))

        total_test_loss = 0.0
        for ids, test_image_path in enumerate(test_image_paths):
            img = load_image(test_image_path)
            img = gray_to_rgb(img)
            img_input = preprocess_input(img)
            img_output = preprocess_output(img)
            batch_index = ids % batch_size
            image_batch_input[batch_index] = img_input

            if batch_index == batch_size - 1:
                total_test_loss += test(model=model, input_batch=image_batch_input,
                                        output=image_batch_input, criterion=criterion) * batch_index

            if ids == len(test_image_paths) - 1:
                total_test_loss += test(model=model, input_batch=image_batch_input[:batch_index + 1],
                                        output=image_batch_input[:batch_index + 1],
                                        criterion=criterion) * batch_index
                print('test loss: %f' %
                      (total_test_loss / len(test_image_paths)))

                if (total_test_loss / len(test_image_paths)) <= prev_test_loss:
                    torch.save(model, "best_" + model_name + ".pt")
                    prev_test_loss = (total_test_loss / len(test_image_paths))


def train_all(dataset_paths, epoch=100, batch_size=256):
    # Train all the models with corresponding datasets.

    # for dataset_path in dataset_paths:
    # print("Original Color")
    # model_resized = ConvAutoencoder()
    # train(model=model_resized, dataset_path=dataset_paths[0], epoch=epoch, batch_size=batch_size, model_name="original")
    print("2 Color")
    model_2 = ConvAutoencoder()
    train(model=model_2, dataset_path=dataset_paths[1], epoch=epoch, batch_size=batch_size, model_name="2")
    # print("4 Color")
    # model_4 = ConvAutoencoder()
    # train(model=model_4, dataset_path=dataset_paths[2], epoch=epoch, batch_size=batch_size, model_name="4")
    # print("8 Color")
    # model_8 = ConvAutoencoder()
    # train(model=model_8, dataset_path=dataset_paths[3], epoch=epoch, batch_size=batch_size, model_name="8")
    # print("16 Color")
    # model_16 = ConvAutoencoder()
    # train(model=model_16, dataset_path=dataset_paths[4], epoch=epoch, batch_size=batch_size, model_name="16")
    # raise NotImplementedError


train_all(roots)

