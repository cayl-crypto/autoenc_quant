import torch
from autoencoder import ConvAutoencoder
from utils import *

model_encoder = ConvAutoencoder()
print(model_encoder)
model_input = torch.randn(1, 3, 64, 64)
from torchvision import datasets, transforms

model_output = model_encoder.forward(model_input)
model_features = model_encoder.extract_feature(model_input)
print("output shape")
print(model_output.shape)
print("features shape")
print(model_features.shape)
print()

preprocess_input = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
tensor2pil = transforms.ToPILImage()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

import numpy as np
import cv2


def color_quantization(image, K=32):
    image = np.array(image)
    # Defining input data for clustering
    data = np.float32(image).reshape((-1, 3))
    # Defining criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # Applying cv2.kmeans function
    ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(image.shape)
    result = Image.fromarray(result)
    return result


def generate_image(model_path, image_path):
    model = torch.load(model_path, map_location=device)
    # model = torch.load('best_original.pt')

    # image_path = 'test_image.jpg'

    image = load_image(image_path)
    image = resize(image)

    # input_tensor = torch.zeros(1, 3, 64, 64).cuda()
    input_tensor = torch.zeros(1, 3, 64, 64)

    input_tensor[0] = preprocess_input(image)

    output = model.forward(input_tensor)
    # output = (output + 1) / 2
    x = tensor2pil((output[0].data + 1) / 2)
    # Quantize image
    # x = color_quantization(x)
    return x


# show_image(generate_image("/home/eva/Desktop/ozkan/Image Caption Generator/autoenc_quant/best_original.pt", "/home/eva/Desktop/ozkan/Image Caption Generator/Resized/val2017/000000000285.jpg"))
show_image(generate_image("/home/eva/Desktop/ozkan/Image Caption Generator/autoenc_quant/best_original.pt",
                          "/home/eva/Desktop/ozkan/Image Caption Generator/Resized/train2017/000000000036.jpg"))
# show_image(generate_image("/home/eva/Desktop/ozkan/Image Caption Generator/autoenc_quant/best_16.pt", "/home/eva/Desktop/ozkan/Image Caption Generator/quant_learning/16/val2017/000000000285.jpg"))
show_image(generate_image("/home/eva/Desktop/ozkan/Image Caption Generator/autoenc_quant/best_2.pt",
                          "/home/eva/Desktop/ozkan/Image Caption Generator/quant_learning/2/train2017/000000000036.jpg"))
show_image(generate_image("/home/eva/Desktop/ozkan/Image Caption Generator/autoenc_quant/best_4.pt",
                          "/home/eva/Desktop/ozkan/Image Caption Generator/quant_learning/4/train2017/000000000036.jpg"))
show_image(generate_image("/home/eva/Desktop/ozkan/Image Caption Generator/autoenc_quant/best_8.pt",
                          "/home/eva/Desktop/ozkan/Image Caption Generator/quant_learning/8/train2017/000000000036.jpg"))
show_image(generate_image("/home/eva/Desktop/ozkan/Image Caption Generator/autoenc_quant/best_16.pt",
                          "/home/eva/Desktop/ozkan/Image Caption Generator/quant_learning/16/train2017/000000000036.jpg"))
show_image(generate_image("/home/eva/Desktop/ozkan/Image Caption Generator/autoenc_quant/best_32.pt",
                          "/home/eva/Desktop/ozkan/Image Caption Generator/quant_learning/32/train2017/000000000036.jpg"))
show_image(generate_image("/home/eva/Desktop/ozkan/Image Caption Generator/autoenc_quant/best_64.pt",
                          "/home/eva/Desktop/ozkan/Image Caption Generator/quant_learning/64/train2017/000000000036.jpg"))
show_image(generate_image("/home/eva/Desktop/ozkan/Image Caption Generator/autoenc_quant/best_128.pt",
                          "/home/eva/Desktop/ozkan/Image Caption Generator/quant_learning/128/train2017/000000000036.jpg"))
# show_image(generate_image("/home/eva/Desktop/ozkan/Image Caption Generator/autoenc_quant/best_8.pt",
# "/home/eva/Desktop/ozkan/Image Caption Generator/quant_learning/8/val2017/000000000776.jpg"))
