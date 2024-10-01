import os
import torch

from PIL import Image
from pathlib import Path
from train import load_model
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader

# from resnet import ResNet50
# from train_classifier import create_data_loaders

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
)


def calculate_tpr_tnr(tp, fn, tn, fp):
    tpr = tp / (tp + fn) if tp + fn > 0 else 0
    tnr = tn / (tn + fp) if tn + fp > 0 else 0
    return tpr * 100, tnr * 100


class CustomImageDataset:
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(self.root_dir)

        self.data = []  # List to store (image_path, label) pairs

        for class_dir in self.classes:
            class_path = os.path.join(self.root_dir, class_dir)
            if not os.path.isdir(class_path):  # Check if it's a directory
                continue
            label = 0 if class_dir == "original" else 1  # 0 for original, 1 for fake

            for file_name in os.listdir(class_path):
                self.data.append((os.path.join(class_path, file_name), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label, image_path


def load_dataset(data_root):
    print(data_root)
    # Create custom datasets for each split
    # train_dataset = CustomImageDataset(
    #     os.path.join(data_root, "train"), transform=transform
    # )
    val_dataset = CustomImageDataset(data_root + "val", transform=transform)
    # test_dataset = CustomImageDataset(
    #     os.path.join(data_root, "test"), transform=transform
    # )
    #
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return valid_loader


# Load all models and their datasets
models = {}
val_data_loaders = {}

for model_name in os.listdir("./trained_models/"):
    if model_name != "model_pre.pt":
        model_path = os.path.join("./trained_models", model_name)
        model = load_model(False)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()  # Set the model to evaluation mode
        extracted_model_name = model_name.replace("resnet_", "").replace(".pt", "")
        models[extracted_model_name] = model
        # Load the validation data loader for this model
        data_prefix = f"/home/nick/ff_crops/{extracted_model_name}_crops/"

        val_data_loader = load_dataset(data_prefix)
        val_data_loaders[model_name] = val_data_loader

        print(f"Loaded model and validation data loader for: {model_name}")

# Cross-testing of each model on all validation datasets
for model_name, model in models.items():
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fun = torch.nn.CrossEntropyLoss()
    for dataset_name, val_loader in val_data_loaders.items():
        positive_indices = []
        negative_indices = []
        tp, tn, fp, fn = 0, 0, 0, 0

        correct = 0
        total = 0
        with torch.no_grad():
            for i, (images, labels, image_paths) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                val_loss = loss_fun(out, labels)
                _, predicted = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for image_idx, (true_label, pred_label) in enumerate(
                    zip(labels, predicted)
                ):
                    if true_label == 1:
                        if pred_label == 1:
                            positive_indices.append(image_paths[image_idx])
                            tp += 1
                        else:
                            fn += 1
                    else:
                        if pred_label == 0:
                            negative_indices.append(image_paths[image_idx])
                            tn += 1
                        else:
                            fp += 1

            accuracy = 100 * correct / total
            tpr, tnr = calculate_tpr_tnr(tp, fn, tn, fp)
        dataset_name = dataset_name.replace("resnet_", "").replace(".pt", "")

        print(f"Model {model_name} on {dataset_name} dataset: {accuracy:.2f}% accuracy")
        print(f"True Positive Rate (TPR): {tpr:.2f}")
        print(f"True Negative Rate (TNR): {tnr:.2f}")
        torch.save(
            positive_indices, f"./tpnr/{model_name}_{dataset_name}_positive_indices.pt"
        )
        torch.save(
            negative_indices, f"./tpnr/{model_name}_{dataset_name}_negative_indices.pt"
        )
