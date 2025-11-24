

import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms




cifar_data = datasets.CIFAR10(root="data",
    train=True,
    download=True,
    transform= transforms.ToTensor()
    )
### Training and Test
def datasets(data = cifar_data):
    training_data_subset, test_subset = random_split(cifar_data, [0.8, 0.2])
    training_images = torch.tensor(cifar_data.data[training_data_subset.indices]).float()/255.0
    training_labels = torch.tensor(cifar_data.targets)[training_data_subset.indices]


    ### Test and Validation
    test_data_subset, validation_data_subset = random_split(test_subset, [0.6, 0.4])
    test_images = torch.tensor(cifar_data.data)[test_data_subset.indices].float()/255.0
    test_label = torch.tensor(cifar_data.targets)[test_data_subset.indices]

    ### Validation
    validation_images = torch.tensor(cifar_data.data)[validation_data_subset.indices].float()/255.0
    validation_labels = torch.tensor(cifar_data.targets)[validation_data_subset.indices]

    batched_training_images = training_images.clone().permute(0, 3, 1, 2)
    batched_validation_images = validation_images.clone().permute(0, 3, 1, 2)

    batch_n = 400 #batch size of 100
    '''
    batch_n = 400
    batch_size = 100
    N = batch_size = 100
    H = 32
    W = 32
    C = 3
    '''
    norm_mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, -1, 1, 1)
    norm_std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, -1, 1, 1)

    normalize_transform = torchvision.transforms.Normalize(norm_mean, norm_std)
    transform_norm = torchvision.transforms.Compose([
        normalize_transform
    ])


    normalized_training_images = transform_norm(batched_training_images)
    normalized_validation_images = transform_norm(batched_validation_images)

    batched_training_images = normalized_training_images.reshape(400, 100, 3, 32, 32)
    batched_training_labels = training_labels.reshape(400, 100)

    batched_validation_images = normalized_validation_images.reshape(40, 100, 3, 32, 32)
    batched_validation_labels = validation_labels.reshape(40, 100)



    return batched_training_images, batched_training_labels, batched_validation_images, batched_validation_labels

