import numpy as np
import pandas as pd
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms, models

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

# ------------------------------------

def create_X_y(df, im_dir, is_save=False, is_transform=False, apply_norm=True, crop_img=False, new_dim=100):
    X = []
    y = []
    counter = 0
    for image_name in df['new_filename'].to_list():
        # print(image_name)
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

    channels, height, width = image.shape

    # This check was added because some images are automatically loaded as grayscale
    if image.shape[0] < 3:
        image = image.expand(3, -1, -1)
    # This check is for images like 18807 that have extra channels with zero information
    if image.shape[0] > 3:
        image = image[0:3, :, :]

    # This is the imagenet normalizer
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

    return image


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


def main():
    # loading the csv file with the data info
    # run prepare_dataframe.py before running this code
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

    train_dataset = SiameseDataset(X_train, y_train, length=X_train.shape[0])
    val_dataset = SiameseDataset(X_val, y_val, length=X_val.shape[0] * 10)
    test_dataset = SiameseDataset(X_test, y_test, length=X_test.shape[0] * 20) #  we want to check many
                                                                               # pairs to prove the model's abilities

    torch.save(train_dataset, f'saved_datasets/train_{NUM_OF_ARTISTS_TRAIN}_i{NUM_IMAGES_PER_ARTIST}_d{IMAGE_DIM}.pt', pickle_protocol=5)
    torch.save(val_dataset, f'saved_datasets/val_{NUM_OF_ARTISTS_VAL}_i{NUM_IMAGES_PER_ARTIST}_d{IMAGE_DIM}.pt', pickle_protocol=5)
    torch.save(test_dataset, f'saved_datasets/test_{NUM_OF_ARTISTS_TEST}_i{NUM_IMAGES_PER_ARTIST}_d{IMAGE_DIM}.pt', pickle_protocol=5)

    print("saved models successfully")

    # loading the datasets
    train_dataset = torch.load(f'saved_datasets/train_{NUM_OF_ARTISTS_TRAIN}_i{NUM_IMAGES_PER_ARTIST}_d{IMAGE_DIM}.pt')
    val_dataset = torch.load(f'saved_datasets/val_{NUM_OF_ARTISTS_VAL}_i{NUM_IMAGES_PER_ARTIST}_d{IMAGE_DIM}.pt')
    test_dataset = torch.load(f'saved_datasets/test_{NUM_OF_ARTISTS_TEST}_i{NUM_IMAGES_PER_ARTIST}_d{IMAGE_DIM}.pt')

    print("loaded successfully")

if __name__ == '__main__':
    main()
