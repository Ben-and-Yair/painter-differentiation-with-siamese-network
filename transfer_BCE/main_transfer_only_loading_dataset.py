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


def create_X_y(df, im_dir, is_save=False):
    X = []
    y = []
    for image_name in df['new_filename'].to_list():
        image = cv2.imread(im_dir+'/'+image_name)
        X.append(image)
        label = int(df.loc[df['new_filename'] == image_name]["artist_code"].iloc[0])
        y.append(label)
        if is_save:
            cv2.imwrite(f'actual_{im_dir}/{image_name}', image)

    return X, y


def transform_images(X, apply_norm=True, crop_img=False, new_dim=100):
    X_res = []
    counter = 0
    for im in X:
        if type(im) is None:
            counter += 1
            continue
        im_res = image_transformer_nn(im, apply_norm=apply_norm, crop_img=crop_img, new_dim=new_dim)
        X_res.append(im_res)

    print("lost ", counter, " images")
    return np.array(X_res)


# taken from here https://www.kaggle.com/code/spyrosrigas/20-painters-classification-with-cnns-and-svms
def image_transformer_nn(image, apply_norm=True, crop_img=True, new_dim=224):
    """
    Args:
        resize_num (int):
            Dimension (pixels) to resize image
        apply_norm (bool):
            Choose whether to apply the normalization or not
        crop_img (bool):
            Choose whether to resize the image into the new_dim size, or crop
            a square from its center, sized new_dim x new_dim
    """
    if crop_img:
        cropper = transforms.CenterCrop(new_dim)
        image = cropper(image)
    # Using transforms.Compose() is another option to perform these sequentially, but
    # let's keep it like this until we find the "final" transformations sequence
    tensoring = transforms.ToTensor()
    image = tensoring(image)  # shape is now (channels, height, width), see next line
    channels, height, width = image.shape

    # This check was added because some images are automatically loaded as grayscale
    if image.shape[0] < 3:
        image = image.expand(3, -1, -1)
    # This check is for images like 18807 that have extra channels with zero information
    if image.shape[0] > 3:
        image = image[0:3, :, :]

    # This is the imagenet normalizer, maybe define our own?
    if apply_norm:
        normalizer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        image = normalizer(image)

    if not crop_img:
        if width < height:
            # Convolutions are invariant to rotations, so we choose to pad everything
            # "down". This means that for landscape images (width > height) no rotation
            # needs to be performed we just pad "down". For vertical images (width < height),
            # in order to perform the "down" padding we have to rotate them first.
            image = image.transpose(1, 2)
        channels, height, width = image.shape
        res_percent = float(new_dim / width)  # done to keep aspect ratio, width is max dim
        height = round(height * res_percent)
        resizer = transforms.Resize((height, new_dim))
        image = resizer(image)
        # Now that the image is resized by keeping aspect ratio, we pad "down"
        padder = transforms.Pad([0, 0, 0, int(new_dim - height)])
        image = padder(image)

    # image = image.numpy().transpose(1, 2, 0)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)

    return image


# originally taken from https://github.com/pytorch/examples/blob/main/siamese_network/main.py#L97
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


def train(model, device, train_loader, optimizer, epoch):
    model.train()

    criterion = nn.BCELoss()

    train_loss_per_batch = []

    for batch_idx, (images_1, images_2, targets) in enumerate(train_loader):
        images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images_1, images_2).squeeze()
        targets = targets.squeeze()
        b_size = outputs.shape[0]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{epoch * len(images_1)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tAverage Loss: {loss.item() / b_size:.6f}')

        train_loss_per_batch.append(loss.item() / b_size)

    return train_loss_per_batch


def test(model, device, test_loader, name):
    model.eval()
    test_loss = 0
    correct = 0

    criterion = nn.BCELoss()

    with torch.no_grad():
        for (images_1, images_2, targets) in test_loader:
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            outputs = model(images_1, images_2).squeeze()
            targets = targets.squeeze()
            test_loss += criterion(outputs, targets).sum().item()  # sum up batch loss
            pred = torch.where(outputs > 0.5, 1, 0)
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\n{name} set: Average loss: {test_loss:.4f},'
          f' Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')

    return test_loss, accuracy


def plot_metric(epochs_for_graph, train_metric, test_metric, metric_name='Metric', label_train='Training', label_test='Validation'):
    plt.plot(epochs_for_graph, train_metric, 'b', label=f'{label_train} {metric_name}')
    plt.plot(epochs_for_graph, test_metric, 'r', label=f'{label_test} {metric_name}')
    plt.title(f'{label_train} and {label_test} {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(f'{label_train} and {label_test} {metric_name}')
    plt.close()


def main():
    # loading the datasets
    train_dataset = torch.load(f'saved_datasets/train_{NUM_OF_ARTISTS_TRAIN}_i{NUM_IMAGES_PER_ARTIST}_d{IMAGE_DIM}.pt')
    val_dataset = torch.load(f'saved_datasets/val_{NUM_OF_ARTISTS_VAL}_i{NUM_IMAGES_PER_ARTIST}_d{IMAGE_DIM}.pt')
    test_dataset = torch.load(f'saved_datasets/test_{NUM_OF_ARTISTS_TEST}_i{NUM_IMAGES_PER_ARTIST}_d{IMAGE_DIM}.pt')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # device config
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print('device: ', device)

    model = SiameseNetwork().to(device)
    print(model)

    # loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate)

    loss_over_batches = []
    loss_over_epochs_train = []
    loss_over_epochs_val = []
    acc_over_epochs_train = []
    acc_over_epochs_val = []

    epochs_for_graph = []

    for epoch in range(num_epochs):
        loss_over_batches += train(model, device, train_loader, optimizer, epoch)

        if epoch % 25 == 0:
            train_l, train_a = test(model, device, train_loader, name='Train')
            val_l, val_a = test(model, device, val_loader, name='Validation')

            loss_over_epochs_train.append(train_l)
            loss_over_epochs_val.append(val_l)
            acc_over_epochs_train.append(train_a)
            acc_over_epochs_val.append(val_a)

            epochs_for_graph.append(epoch)

            plot_metric(epochs_for_graph, loss_over_epochs_train, loss_over_epochs_val, metric_name='Loss')
            plot_metric(epochs_for_graph, acc_over_epochs_train, acc_over_epochs_val, metric_name='Accuracy')

        if epoch % 100 == 0:
            path_for_model = MODEL_SAVE_PATH + '/model_' + str(epoch) + '.pth'
            torch.save(model.state_dict(), path_for_model)

    path_for_model = MODEL_SAVE_PATH + '/model_' + str(num_epochs) + '.pth'
    torch.save(model.state_dict(), path_for_model)

    test(model, device, test_loader, name='Test')


if __name__ == '__main__':
    main()
