import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from models import model_selection
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from xception import xception

from dataset.transform import CustomImageDataset

torch.cuda.empty_cache()
transform = transforms.Compose(
    [
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
)


# Define dataset paths


def load_dataset(data_root):
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

    batch_size = 32  # Adjust as needed
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader


def load_model(imagenet, pretrained):
    train_losses = []
    validation_accuracies = []
    test_accuracies = []
    if imagenet:
        model, *_ = model_selection(modelname="xception", num_out_classes=2)
    elif not imagenet:
        model = xception(
            num_classes=2, pretrained="imagenet"
        )  # Adjust num_classes as needed
    if pretrained:
        if imagenet:
            checkpoint = torch.load("model_pre.pt")
            loader = "model_pre.pt"
        else:
            checkpoint = torch.load("model.pt")
            loader = "model.pt"
        train_losses += torch.load(loader)["train_losses"]
        validation_accuracies += torch.load(loader)["validation_accuracies"]
        test_accuracies += torch.load(loader)["test_accuracies"]
        model.load_state_dict(checkpoint["model_state_dict"])
    return model, train_losses, validation_accuracies, test_accuracies


# Training and Validation for a single epoch


def model_train(model, train_loader, device, train_losses):
    print(f"Epoch [1/1]")  # Run for a single epoch
    train_losses = []
    running_loss = 0.0
    model = model.to(device)
    model.train()
    # Use tqdm to create a progress bar
    data_loader = tqdm(train_loader)
    for i, (inputs, labels) in enumerate(data_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 100 == 99:
            data_loader.set_description(f"[1, {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0
        train_losses.append(running_loss / len(train_loader))
        return train_losses


def model_validate(model, valid_loader, device):
    validation_accuracies = []
    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    validation_accuracy = 100 * correct / total

    print(f"Validation accuracy after Epoch 1: {validation_accuracy:.2f}%")
    validation_accuracies.append(validation_accuracy)
    return validation_accuracies


def model_test(model, test_loader, device):
    # Test the model on the test set
    test_accuracies = []
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Test accuracy after Epoch 1: {test_accuracy:.2f}%")
    test_accuracies.append(test_accuracy)


# Save the model state if validation accuracy is better
# Save training statistics
if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--imagenet", "-p", type=str)
    p.add_argument("--pretrained", "-p", type=str)
    p.add_argument("--model_path", "-mi", type=str, default=None)
    args = p.parse_args()

    data_root = "/users/nick/faceforensics_crops/"  # the root folder containing 'real' and 'fake' subfolders
    train_loader, valid_loader, test_loader = load_dataset(data_root)
    model, train_losses, validation_accuracies, test_accuracies = load_model(
        args.imagenet, args.pretrained
    )
    optimizer = optim.adam(model.parameters(), lr=0.001)
    criterion = nn.crossentropyloss()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model_train(model, train_loader, device)
    model_validate(model, valid_loader, device)
    model_test(model, test_loader, device)
    # save the model and training results after the single epoch
    torch.save(
        {
            "epoch": model.load_state_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "validation_accuracies": validation_accuracies,
            "test_accuracies": test_accuracies,  # Save test accuracies
        },
        args.model_path,
    )
    print("Finished Training for a single epoch")
