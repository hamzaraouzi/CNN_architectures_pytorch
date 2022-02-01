from torchvision import datasets, transforms
import torch
import config
import os
import random
from shutil import copyfile


def prepare_datasets():
    subdirs = ['train/', 'test/']
    for subdir in subdirs:
        # create label subdirectories
        labeldirs = ['airplane/', 'car/', 'cat/', 'dog/', 'flower/', 'fruit/', 'motorbike/',
                     'person/']
        for labldir in labeldirs:
            newdir = config.DATASET_HOME + subdir + labldir
            os.makedirs(newdir, exist_ok=True)

    val_ratio = 0.25
    # copy training dataset images into subdirectories

    for dir in os.listdir('natural_images/'):

        for file in os.listdir('natural_images/' + dir):
            src = 'natural_images/' + dir + '/' + file
            if random.random() < val_ratio:
                dest = config.DATASET_HOME + 'test/' + dir + '/' + file
            else:
                dest = config.DATASET_HOME + 'train/' + dir + '/' + file

            copyfile(src, dest)



def get_loaders():

    prepare_datasets()

    train_transforms = transforms.Compose([
        transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    test_transforms = transforms.Compose([
        transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    train_folder = os.path.join(config.DATASET_HOME, "train")
    test_folder = os.path.join(config.DATASET_HOME, "test")
    train_data = datasets.ImageFolder(train_folder, transform=train_transforms)
    test_data = datasets.ImageFolder(test_folder, transform=test_transforms)

    # Data Loading
    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=config.BATCH_SIZE)
    testloader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=config.BATCH_SIZE)

    return trainloader, testloader