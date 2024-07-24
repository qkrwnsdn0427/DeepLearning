import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
# from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from model import BaseModel
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch.nn as nn # edited

class ImageDataset(Dataset):

    def __init__(self, root_dir, transform=None, fmt=':04d', extension='.jpg'):
        self.root_dir = root_dir
        self.fmtstr = '{' + fmt + '}' + extension
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.fmtstr.format(idx)
        img_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        data = self.transform(img)
        return data

# Define TTA transforms#
tta_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
])#

def inference_with_tta(args, data_loader, model, tta_steps=10):
    """ model inference with TTA """
    model.eval()
    final_preds = []
    
    with torch.no_grad():###
        pbar = tqdm(data_loader)
        for images in pbar:
            images = images.to(args.device)
            tta_preds = []
            for _ in range(tta_steps):
                # Apply TTA transforms and make predictions
                augmented_images = tta_transforms(images)
                outputs = model(augmented_images).softmax(dim=-1)
                tta_preds.append(outputs.cpu())
            # Average the predictions across the TTA steps
            tta_preds = torch.mean(torch.stack(tta_preds), dim=0)
            final_preds.append(tta_preds.argmax(dim=-1).numpy())
    
    # Concatenate all batches
    final_preds = np.concatenate(final_preds)
    return final_preds###


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2023 DL Term Project')
    parser.add_argument('--load-model', default='checkpoints/model.pth', help="Model's state_dict")
    parser.add_argument('--batch-size', default=16, help='test loader batch size')
    parser.add_argument('--dataset', default='test_images/', help='image dataset directory')

    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    # torchvision model
    num_classes = 10
    model = efficientnet_b3()
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    
    model.load_state_dict(torch.load(args.load_model, map_location=device))
    model.to(device)
    
    # load dataset in test image folder
    # you may need to edit transform
    # load dataset in test image folder with TTA transforms
    ##
    test_data = ImageDataset(args.dataset, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Include TTA transforms during dataset creation if you want to visualize the augmented images
    ]))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)
    ##

    # write model inference
    preds = inference_with_tta(args, test_loader, model)
        
    with open('result.txt', 'w') as f:
        f.writelines('\n'.join(map(str, preds)))