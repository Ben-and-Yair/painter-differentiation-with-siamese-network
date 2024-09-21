# painter-differentiation-with-siamese-network
# Artist Classification Using Deep Learning

This project was developed as part of a deep learning course at the University of Haifa. The goal of the project was to build and compare two types of neural networks—a Triplet Loss-based network and a Binary Cross Entropy (BCE) network—to differentiate between different artists, including those who were not seen during training. We used the **Painters by Numbers** dataset from Kaggle as our primary data source.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Triplet Loss Network](#triplet-loss-network)
  - [Binary Cross Entropy Network](#binary-cross-entropy-network)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)
- [References](#references)

## Project Overview
The primary objective of this project was to build a system that could recognize and differentiate between different artists based on their paintings. We implemented and trained two neural networks:
- **Triplet Loss-based network**: This network learns embeddings for images such that images from the same artist are closer together, while images from different artists are further apart.
- **Binary Cross Entropy (BCE) network**: This network performs artist classification directly, learning to classify whether a given image belongs to a specific artist or not.

Both networks were tested on previously unseen artists to evaluate their ability to generalize to new data.

## Dataset
The **Painters by Numbers** dataset from Kaggle contains over 100,000 images of paintings by various artists, making it an ideal dataset for this task. The dataset includes both labels for the artist and metadata about the paintings. You can download the dataset from [Kaggle](https://www.kaggle.com/c/painter-by-numbers/data).

## Methodology

### Triplet Loss Network
The Triplet Loss network aims to learn a mapping from paintings to a feature space where paintings by the same artist are closer to each other, and paintings by different artists are far apart. The network processes image triplets consisting of:
- **Anchor**: A painting from a given artist.
- **Positive**: Another painting from the same artist.
- **Negative**: A painting from a different artist.

We used a pre-trained Convolutional Neural Network (CNN) as the backbone for our embedding model, followed by a fully connected layer for dimensionality reduction.

### Binary Cross Entropy Network
The BCE network is a standard classification network where the objective is to classify whether a painting belongs to a particular artist or not. We trained the network with one-hot encoded labels for artists and used Binary Cross Entropy loss. This network was evaluated both on artists included in the training set and on unseen artists during the testing phase.

## Results
- **Triplet Loss Network**: Demonstrated strong performance in clustering paintings by the same artist, showing generalization capabilities for unseen artists.
- **BCE Network**: Achieved competitive accuracy for artist classification on the training artists but struggled to generalize to unseen artists.

Detailed results and evaluation metrics can be found in the report included in this repository.

## Requirements
- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Kaggle API (for dataset downloading)
