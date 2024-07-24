import argparse

import numpy as np
from tqdm import tqdm
from utils._utils import make_data_loader
from model import BaseModel

import torch
# from torchvision.models import resnet18, ResNet18_Weights
# 다른 모델
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights#
from torchvision import transforms#

def acc(pred,label):
    pred = pred.argmax(dim=-1)
    return torch.sum(pred == label).item()


# Define TTA transforms#
tta_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
])#


def validate(model, data_loader, device):#
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total#


# def validate_with_tta(model, data_loader, device, tta_steps=10):
#     model.eval()
#     final_preds = []
#     final_labels = []
#     with torch.no_grad():
#         for images, labels in data_loader:
#             images = images.to(device)
#             labels = labels.to(device).cpu().numpy()
#             tta_preds = []
#             for _ in range(tta_steps):
#                 # Apply TTA transforms and make predictions
#                 augmented_images = tta_transforms(images)
#                 outputs = model(augmented_images).softmax(dim=-1)
#                 tta_preds.append(outputs.cpu())
#             # Average the predictions across the TTA steps
#             tta_preds = torch.mean(torch.stack(tta_preds), dim=0)
#             final_preds.append(tta_preds.argmax(dim=-1).numpy())
#             final_labels.append(labels)
#     # Concatenate all batches
#     final_preds = np.concatenate(final_preds)
#     final_labels = np.concatenate(final_labels)
#     # Calculate accuracy
#     correct = (final_preds == final_labels).sum()
#     total = final_labels.shape[0]
#     return correct / total



def train(args, data_loader, model):#, start_epoch
    """
    TODO: Change the training code as you need. (e.g. different optimizer, different loss function, etc.)
            You can add validation code. -> This will increase the accuracy.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    for epoch in range(args.epochs):#start_epoch, 
        train_losses = [] 
        train_acc = 0.0
        total=0
        print(f"[Epoch {epoch+1} / {args.epochs}]")
        
        model.train()
        pbar = tqdm(data_loader)
        for i, (x, y) in enumerate(pbar):
            image = x.to(args.device)
            label = y.to(args.device)          
            optimizer.zero_grad()

            output = model(image)
            
            label = label.squeeze()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            total += label.size(0)

            train_acc += acc(output, label)

        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = train_acc/total
        
        print(f'Epoch {epoch+1}') 
        print(f'train_loss : {epoch_train_loss}')
        print('train_accuracy : {:.3f}'.format(epoch_train_acc*100))

        # Save the model and validate with TTA every 4 epochs#
        if (epoch + 1) % 1 == 0:
            general_accuracy = validate(model, val_loader, args.device)
            # tta_accuracy = validate_with_tta(model, val_loader, args.device)

            print('Validation accuracy after epoch {}: {:.3f}%'.format(epoch + 1, general_accuracy * 100))
            # print('Validation accuracy with TTA after epoch {}: {:.3f}%'.format(epoch + 1, tta_accuracy * 100))
            

            
            # Save the model's state_dict
            torch.save(model.state_dict(), f'{args.save_path}/model_epoch_{epoch+1}.pth')#

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='2023 DL Term Project')
    parser.add_argument('--save-path', default='checkpoints/', help="Model's state_dict")
    parser.add_argument('--data', default='data/', type=str, help='data folder')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    """
    TODO: You can change the hyperparameters as you wish.
            (e.g. change epochs etc.)
    """
    
    # hyperparameters
    args.epochs = 40#
    args.learning_rate = 0.001#
    args.batch_size = 16#

    # check settings
    print("==============================")
    print("Save path:", args.save_path)
    print('Using Device:', device)
    print('Number of usable GPUs:', torch.cuda.device_count())
    
    # Print Hyperparameter
    print("Batch_size:", args.batch_size)
    print("learning_rate:", args.learning_rate)
    print("Epochs:", args.epochs)
    print("==============================")
    
    # Make Data loader and Model
    train_loader, val_loader = make_data_loader(args)

    # custom model 기존 모델
    # model = BaseModel()

    # torchvision model
    #model = resnet18(weights=ResNet18_Weights)
    # Initialize the EfficientNet_B3 model with pre-trained weights
    weights = EfficientNet_B3_Weights.DEFAULT#
    model = efficientnet_b3(weights=weights)#

    # Adjust the number of output features to the number of classes in your dataset
    num_classes = 10
    #num_features = model.fc.in_features # edited
    #model.fc = nn.Linear(num_features, num_classes) # edited

    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)#

    # 모델 상태 불러오기
    # 체크포인트 로드
    # checkpoint = torch.load("/content/drive/MyDrive/Term_Project/checkpoints/model_epoch_27.pth")
    # model.load_state_dict(checkpoint)

    model.to(device)
    print(model)

    # Training The Model
    train(args, train_loader, model)#, start_epoch=27