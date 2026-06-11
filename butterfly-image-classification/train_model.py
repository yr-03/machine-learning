from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import torch
import torch.optim as optim
import torch.nn as nn

import torchvision.transforms as transforms

from dataloader import CustomImageDataset
from model import NeuralNet

def main(learning_rate=0.0003, epochs=75):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.enabled = False

    cnn_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5), # Flips the butterfly orientation
        transforms.RandomRotation(degrees=15),   # Simulates varying camera angles
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # Simulates outdoor shadows/lighting
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    SCRIPT_DIR = Path(__file__).resolve().parent

    CSV_PATH = SCRIPT_DIR / "data" / "Training_set.csv"
    IMG_DIR = SCRIPT_DIR / "data" / "train"

    full_df = pd.read_csv(CSV_PATH)

    train_df, test_df = train_test_split(
        full_df,
        test_size=0.2,
        random_state=42,
        stratify=full_df['label']
    )

    train_dataset = CustomImageDataset(dataframe=train_df, img_dir=str(IMG_DIR), transform=cnn_transforms)
    test_dataset = CustomImageDataset(dataframe=test_df, img_dir=str(IMG_DIR), transform=cnn_transforms)

    BATCH_SIZE = 256

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    net = NeuralNet().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimiser = optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(f'\n--- Starting Training Epoch {epoch} ---')

        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimiser.zero_grad()
            outputs = net(inputs)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimiser.step()

            running_loss += loss.item()

        print(f'Loss: {running_loss / len(train_loader):.4f}')

    torch.save(net.state_dict(), 'trained_net.pth')

    net.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in train_loader:
            images, labels = data
        
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = net(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Train Accuracy: {accuracy}%')


    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
        
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = net(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy}%')

    

if __name__ == '__main__':
    main(
        learning_rate=0.0003,
        epochs=60
    )