from torchvision import datasets, transforms
import torch

from torch.utils.data import DataLoader, Subset, random_split
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomRotation, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import RandomGrayscale

# to split test_data always same
torch.manual_seed(1004)

import torch.nn as nn
from torch.utils.data import DataLoader


# Define your scales
scales = [32, 128, 224]

# Custom transform for random resizing
class RandomResizeTransform:
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, img):
        # Randomly choose a scale
        scale_size = random.choice(self.scales)
        return transforms.Resize((scale_size, scale_size))(img)


train_transforms = transforms.Compose([
        RandomResizeTransform(scales),
        transforms.RandomResizedCrop(224),  # EfficientNet expects 224x224 inputs
        transforms.RandomHorizontalFlip(),  # Randomly flip the images horizontally
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Randomly change the brightness, contrast, saturation and hue
        RandomGrayscale(p=0.2),  # 50% chance to convert to grayscale
        transforms.RandomRotation(15),  # Randomly rotate the images by +/- 15 degrees
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet mean and std
    ])

val_transforms = transforms.Compose([
        transforms.Resize(256),  # Resize the images to 256x256
        transforms.CenterCrop(224),  # Crop the images to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def make_data_loader(args):
    
    # Get Dataset
    dataset = datasets.ImageFolder(args.data)
    
    # Split the dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Apply the transforms to the training and validation datasets using a Lambda transform
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transforms

    # Create data loaders for the training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader