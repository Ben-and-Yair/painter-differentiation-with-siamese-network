import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F

import random

# ---------- CONFIGURATIONS ----------
MODEL_SAVE_PATH = 'saved_models'

NUM_OF_ARTISTS_TRAIN = 800
NUM_OF_ARTISTS_VAL = 100
NUM_OF_ARTISTS_TEST = 100

NUM_OF_ARTISTS = NUM_OF_ARTISTS_TRAIN + NUM_OF_ARTISTS_VAL + NUM_OF_ARTISTS_TEST

IMAGE_DIM = 200
# images for each artist
NUM_IMAGES_PER_ARTIST = 20

# ----- Hyperparameters -----
learning_rate = 0.0001
num_epochs = 800
batch_size = 32
# ---------------------------
PATH_TO_DATASET = (f'saved_datasets/test_{NUM_OF_ARTISTS_TEST}_i{NUM_IMAGES_PER_ARTIST}_d{IMAGE_DIM}.pt')
PATH_TO_MODEL = f'{MODEL_SAVE_PATH}/model_700.pth'
# ------------------------------------


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # get resnet model
        self.resnet = torchvision.models.resnet18(weights='DEFAULT')
        self.fc_in_features = self.resnet.fc.in_features

        # freeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        print('self.fc_in_features: ', self.fc_in_features)

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

        self.sigmoid = nn.Sigmoid()

        # initialize the weights
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)

        return output


class SiameseDataset(Dataset):
    def __init__(self, X, y, length):
        super(SiameseDataset, self).__init__()

        self.X = X
        self.y = y
        self.length = length

        # mapping unique labels to new values
        unique_labels = np.unique(y)
        label_map = {label: i for i, label in enumerate(unique_labels)}
        y_new = [label_map[label] for label in y]

        self.num_of_labels = len(unique_labels)
        print("num of labels: ", self.num_of_labels)

        # group X according to labels in y_new
        self.grouped_X = {}
        for label, array in zip(y_new, X):
            if label not in  self.grouped_X:
                self.grouped_X[label] = [array]
            else:
                self.grouped_X[label].append(array)

        # convert lists of arrays to arrays of arrays
        for label, arrays in self.grouped_X.items():
            self.grouped_X[label] = np.array(arrays)


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # pick some random class for the first image
        selected_class = random.randint(0, self.num_of_labels - 1)

        # pick a random index for the first image in the grouped indices based of the label
        # of the class
        random_index_1 = random.randint(0, self.grouped_X[selected_class].shape[0] - 1)

        # get the first image
        image_1 = self.grouped_X[selected_class][random_index_1]

        # same class
        if index % 2 == 0:
            # pick a random index for the second image
            random_index_2 = random.randint(0, self.grouped_X[selected_class].shape[0] - 1)

            # ensure that the index of the second image isn't the same as the first image
            while random_index_2 == random_index_1:
                random_index_2 = random.randint(0, self.grouped_X[selected_class].shape[0] - 1)

            # pick the index to get the second image
            image_2 = self.grouped_X[selected_class][random_index_2]

            # set the label for this example to be positive (1)
            target = torch.tensor(1, dtype=torch.float)

        # different class
        else:
            # pick a random class
            other_selected_class = random.randint(0, self.num_of_labels - 1)

            # ensure that the class of the second image isn't the same as the first image
            while other_selected_class == selected_class:
                other_selected_class = random.randint(0, self.num_of_labels - 1)

            # pick a random index for the second image in the grouped indices based of the label
            # of the class
            random_index_2 = random.randint(0, self.grouped_X[other_selected_class].shape[0] - 1)

            # pick the index to get the second image
            image_2 = self.grouped_X[other_selected_class][random_index_2]

            # set the label for this example to be negative (0)
            target = torch.tensor(0, dtype=torch.float)

        return image_1, image_2, target


def test(model, device, test_loader, name):
    model.eval()
    test_loss = 0
    correct = 0

    criterion = nn.BCELoss()

    stop_after = int(len(test_loader))
    num_of_tests = stop_after * batch_size
    i = 0
    with torch.no_grad():
        for (images_1, images_2, targets) in test_loader:
            if i == stop_after:
                break
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            outputs = model(images_1, images_2).squeeze()
            targets = targets.squeeze()
            test_loss += criterion(outputs, targets).sum().item()  # sum up batch loss
            pred = torch.where(outputs > 0.5, 1, 0)
            correct += pred.eq(targets.view_as(pred)).sum().item()
            i += 1

    test_loss /= num_of_tests
    accuracy = 100. * correct / num_of_tests

    print(f'\n{name} set: Average loss: {test_loss:.4f},'
          f' Accuracy: {correct}/{num_of_tests} ({accuracy:.0f}%)\n')

    return test_loss, accuracy


def main():
    # loading the datasets
    test_dataset = torch.load(PATH_TO_DATASET)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # device config
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print('device: ', device)

    model = SiameseNetwork().to(device)
    print(model)
    model.load_state_dict(torch.load(PATH_TO_MODEL))
    model.eval()
    test(model, device, test_loader, name='Test')


if __name__ == '__main__':
    main()
