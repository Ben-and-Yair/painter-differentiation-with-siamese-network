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

from PIL import Image


import random

# ---------- CONFIGURATIONS ----------
DATA_PATH = r'C:\Users\YairSlobodin\DLFinalProject\pythonProject\Data'

MODEL_SAVE_PATH = 'saved_models'

NUM_OF_ARTISTS_TRAIN = 800
NUM_OF_ARTISTS_VAL = 100
NUM_OF_ARTISTS_TEST = 100

NUM_OF_ARTISTS = NUM_OF_ARTISTS_TRAIN + NUM_OF_ARTISTS_VAL + NUM_OF_ARTISTS_TEST

IMAGE_DIM = 200
# images for each artist
NUM_IMAGES_PER_ARTIST = 20

# ----- Hyperparameters -----
learning_rate = 0.001
num_epochs = 500
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

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.fc_in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )

        # initialize the weights
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output



def create_X_y(df, im_dir, is_save=False, is_transform=False, apply_norm=True, crop_img=False, new_dim=100):
    X = []
    y = []
    counter = 0
    for image_name in df['new_filename'].to_list():
        if counter % 500 == 0:
            print(counter)
        counter += 1
        image = cv2.imread(im_dir+'/'+image_name)

        # checking if the image is corrupt
        if image_name.endswith('.jpg'):
            try:
                img = Image.open(im_dir+'/'+image_name)  # open the image file
                img.verify()  # verify that it is, in fact an image

                if is_transform:
                    image = image_transformer_nn(image, apply_norm=apply_norm, crop_img=crop_img, new_dim=new_dim)

                X.append(image)
                label = int(df.loc[df['new_filename'] == image_name]["artist_code"].iloc[0])
                y.append(label)
                if is_save:
                    cv2.imwrite(f'actual_{im_dir}/{image_name}', image)
            except :
                print('Bad file:', image_name)  # print out the names of corrupt files

    return X, y


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
    # image = torch.from_numpy(image).float()
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

class SiameseDatasetTriplets(Dataset):
    def __init__(self, X, y, length):
        super(SiameseDatasetTriplets, self).__init__()

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
        # pick some random class for the anchor image
        selected_class = random.randint(0, self.num_of_labels - 1)

        # pick a random index for the anchor image in the grouped indices based of the label
        # of the class
        random_index_1 = random.randint(0, self.grouped_X[selected_class].shape[0] - 1)

        # get the anchor image
        anchor = self.grouped_X[selected_class][random_index_1]

        # pick a random index for the positive image
        random_index_2 = random.randint(0, self.grouped_X[selected_class].shape[0] - 1)

        # ensure that the index of the second image isn't the same as the anchor image
        while random_index_2 == random_index_1:
            random_index_2 = random.randint(0, self.grouped_X[selected_class].shape[0] - 1)

        # pick the index to get the positive image
        positive = self.grouped_X[selected_class][random_index_2]

        # pick a random class
        other_selected_class = random.randint(0, self.num_of_labels - 1)

        # ensure that the class of the negative image isn't the same as the anchor image
        while other_selected_class == selected_class:
            other_selected_class = random.randint(0, self.num_of_labels - 1)

        # pick a random index for the negative image in the grouped indices based of the label
        # of the class
        random_index_2 = random.randint(0, self.grouped_X[other_selected_class].shape[0] - 1)

        # pick the index to get the negative image
        negative = self.grouped_X[other_selected_class][random_index_2]

        return anchor, positive, negative


def train(model, device, train_loader, optimizer, epoch):
    model.train()

    criterion = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)

    train_loss_per_batch = []

    overall = 0
    loss_so_far = 0

    for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        optimizer.zero_grad()
        output_a = model(anchor).squeeze()
        output_p = model(positive).squeeze()
        output_n = model(negative).squeeze()

        loss = criterion(output_a, output_p, output_n)
        loss.backward()
        optimizer.step()

        overall += output_a.shape[0]
        loss_so_far += loss.item()

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(anchor)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]'
                  f'\tLoss: { loss_so_far / overall:.6f}')

        train_loss_per_batch.append(loss.item())

    return train_loss_per_batch


def test(model, device, test_loader, name):
    model.eval()
    test_loss = 0
    correct = 0
    overall = 0

    criterion = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)

    with torch.no_grad():
        for batch_idx, (anchor, positive, negative) in enumerate(test_loader):
            # get feature vector of each image
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            output_a = model(anchor).squeeze()
            output_p = model(positive).squeeze()
            output_n = model(negative).squeeze()

            loss = criterion(output_a, output_p, output_n)
            test_loss += loss.sum().item()  # sum up batch loss

            # calculate the Euclidean distance between the anchor, positive image
            # and the distance between the anchor and negative image
            ap_distance = ((output_a - output_p) ** 2).sum(axis=1)
            an_distance = ((output_a - output_n) ** 2).sum(axis=1)

            # if the distance between the anchor and positive images is smaller than the distance
            # between the anchor and negative image then it is considered a successful prediction
            subtract = ap_distance - an_distance
            pred = torch.where(subtract < 0, 1, 0)


            correct += torch.sum(pred).detach().cpu()
            overall += pred.shape[0]

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / overall

    print(f'\n{name} set: Average loss: {test_loss:.4f},'
          f' Accuracy: {correct}/{overall} ({accuracy:.2f}%)\n')

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
    # loading the csv file with the data info
    df = pd.read_csv('all_data_info.csv')

    print(df.columns)
    print(df.info())
    # finding the top NUM_OF_ARTISTS with the most paintings in the dataset
    paintings = df['artist'].value_counts().head(NUM_OF_ARTISTS)
    artists = paintings.index.to_list()
    df = df.loc[df['artist'].isin(artists)]
    print(artists)
    for artist in artists:
        print(f'{artist} has {paintings[artist]} paintings in the dataset')


    artists_codes = {artists[i]: i for i in range(len(artists))}
    df['artist_code'] = df['artist'].map(artists_codes)

    artists_train = artists[:NUM_OF_ARTISTS_TRAIN]
    artists_val = artists[NUM_OF_ARTISTS_TRAIN: NUM_OF_ARTISTS_VAL + NUM_OF_ARTISTS_TRAIN]
    artists_test = artists[-NUM_OF_ARTISTS_TEST:]

    print('\n')
    print(artists_train)
    print(artists_val)
    print(artists_test)
    print('\n')

    # df.to_csv('actual_df.csv')

    # samples rows from each group
    def sample_images(group):
        return group.sample(n=min(NUM_IMAGES_PER_ARTIST, len(group)))

    subsets = df.groupby('artist')

    train_df_list = []
    val_df_list = []
    test_df_list = []

    for group_name, group_df in subsets:
        group_df_sampled = sample_images(group_df)
        if group_name in artists_train:
            train_df_list.append(group_df_sampled)
        elif group_name in artists_val:
            val_df_list.append(group_df_sampled)
        elif group_name in artists_test:
            test_df_list.append(group_df_sampled)
        else:
            print("artist ", group_name, " not in any dataset, fix it")

    df_train = pd.concat(train_df_list)
    df_val = pd.concat(val_df_list)
    df_test = pd.concat(test_df_list)

    # loading the images and creating X and y
    is_transform_X = True
    is_crop_image = False
    is_normalize_image = True
    X_train, y_train = create_X_y(df_train, DATA_PATH, is_save=False, is_transform=is_transform_X,
                                  apply_norm=is_normalize_image, crop_img=is_crop_image, new_dim=IMAGE_DIM)
    X_val, y_val = create_X_y(df_val, DATA_PATH, is_save=False, is_transform=is_transform_X,
                              apply_norm=is_normalize_image, crop_img=is_crop_image, new_dim=IMAGE_DIM)
    X_test, y_test = create_X_y(df_test, DATA_PATH, is_save=False, is_transform=is_transform_X,
                                apply_norm=is_normalize_image, crop_img=is_crop_image, new_dim=IMAGE_DIM)

    if is_transform_X:
        X_train = np.array(X_train)
        X_val = np.array(X_val)
        X_test = np.array(X_test)

        print("X_train after transform: ", X_train.shape)
        print("X_val after transform: ", X_val.shape)
        print("X_test after transform: ", X_test.shape)
    else:
        print(len(X_train), len(y_train))
        print(len(X_val), len(y_val))
        print(len(X_test), len(y_test))

    train_dataset = SiameseDatasetTriplets(X_train, y_train, length=X_train.shape[0])
    val_dataset = SiameseDatasetTriplets(X_val, y_val, length=X_val.shape[0] * 10)
    test_dataset = SiameseDatasetTriplets(X_test, y_test, length=X_test.shape[0] * 10)

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
