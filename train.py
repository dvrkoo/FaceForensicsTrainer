import os
from datetime import datetime
from pathlib import Path

import numpy as np
import argparse
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from dataset.transform import CustomImageDataset, CustomImageDatasetFromFreq
from network.models import model_selection

if torch.cuda.is_available():
    torch.cuda.empty_cache()
else:
    torch.mps.empty_cache()

torch.multiprocessing.set_sharing_strategy("file_system")
BATCH_SIZE = 32
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
)


def load_dataset(data_root):
    print(data_root)
    # Create custom datasets for each split
    train_dataset = CustomImageDataset(
        os.path.join(data_root, "train"), transform=transform
    )
    val_dataset = CustomImageDataset(
        os.path.join(data_root, "val"), transform=transform
    )
    test_dataset = CustomImageDataset(
        os.path.join(data_root, "test"), transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, valid_loader, test_loader


def load_model(pretrained):
    model, *_ = model_selection(modelname="resnet50", num_out_classes=2)
    if pretrained:
        print("Loading Model")
        checkpoint = torch.load("./model_pre.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
    return model


# Training and Validation for a single epoch


def train_one_epoch(model, optimizer, criterion, loader, epoch, device, writer):
    print(f"train epoch {epoch}")  # Run for a single epoch
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    # Use tqdm to create a progress bar
    for i, (images, labels) in tqdm(
        enumerate(loader), desc=f"training epoch: {epoch}", total=len(loader)
    ):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        writer.add_scalar("train/Minibatch Loss", loss.item(), i + epoch * len(loader))

    accuracy = 100 * correct / total
    writer.add_scalar("train/acc", accuracy, epoch)
    writer.add_scalar("train/Average Loss", running_loss / len(loader), epoch)


@torch.no_grad()
def evaluate(model, criterion, loader, epoch, device, writer, mode):
    print(f"{mode} epoch {epoch}")
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in tqdm(
            enumerate(loader), desc=f"evaluating {mode}", total=len(loader)
        ):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            writer.add_scalar(
                f"{mode}/Minibatch Loss", loss.item(), i + epoch * len(loader)
            )

    accuracy = 100 * correct / total
    writer.add_scalar(f"{mode}/acc", accuracy, epoch)
    writer.add_scalar(f"{mode}/Average Loss", running_loss / len(loader), epoch)


# Save the model state if validation accuracy is better
# Save training statistics
if __name__ == "__main__":
    torch.cuda.empty_cache()
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction)
    p.add_argument("--model_path", "-mp", type=str, default=None)
    p.add_argument("--epochs", "-e", type=int, default=18)
    p.add_argument("--data_root", "-dr", type=str, default=None)

    args = p.parse_args()
    data_root = args.data_root
    train_loader, val_loader, test_loader = load_dataset(data_root)
    model = load_model(args.pretrained)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    writer = SummaryWriter(
        "./SummaryWriter/Resnet_faceshifter" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    print(len(train_loader))
    # move model to cuda
    model.to(device)
    # save the model if an exception occur
    try:
        for epoch in range(args.epochs):
            train_one_epoch(
                model, optimizer, criterion, train_loader, epoch, device, writer
            )
            evaluate(model, criterion, val_loader, epoch, device, writer, mode="val")
            evaluate(model, criterion, test_loader, epoch, device, writer, mode="test")
            torch.cuda.empty_cache()
    except Exception as e:
        print(e)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
            },
            args.model_path,
        )
        print("Model saved")
        writer.close()
        exit(0)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        args.model_path,
    )
    print("Model saved")
    print("Finished Training for a single epoch")
    writer.close()
