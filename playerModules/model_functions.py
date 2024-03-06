import cv2
import dlib
import torch
import torch.nn as nn
import os
import sys
from PIL import Image

from dataset.transform import xception_default_data_transforms
from network.models import model_selection


def get_resource_path(relative_path):
    """Returns the absolute path to a resource, works for development and PyInstaller bundles."""
    if getattr(sys, "frozen", False):  # Check if bundled by PyInstaller
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")  # Current directory if not bundled
    return os.path.join(base_path, relative_path)


print("Cuda available: ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

model_names = [
    "faceswap",
    "deepfake",
    "neuraltextures",
    "face2face",
    "faceshifter",
]


def load_models():
    models = []
    for model_name in model_names:
        model, *_ = model_selection(modelname="xception", num_out_classes=2)
        print(f"Loading {model_name} Model")
        checkpoint = torch.load(
            get_resource_path(f"./trained_models/{model_name}.pt"), map_location="cpu"
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()
        models.append(model)

        detector = dlib.get_frontal_face_detector()

    return models, detector


def convert_color_space(frame):
    # Convert the frame to RGB if it's not already in that color space
    if frame.shape[-1] == 1:  # If the frame is grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[-1] == 3:  # If the frame is in BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def get_boundingbox(face, frame, scale=1.3):
    height, width = frame.shape[:2]
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def detect_faces(frame, detector):
    # Convert the BGR image to RGB explicitly
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the frame using dlib
    return detector(gray, 1)


def preprocess_input(face_roi):
    image = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    preprocess = xception_default_data_transforms["test"]
    preprocessed_image = preprocess(Image.fromarray(image))
    preprocessed_image = preprocessed_image.unsqueeze(0)
    preprocessed_image = preprocessed_image.to(device)
    return preprocessed_image


def predict_with_selected_model(input_tensor, model, post_fuction=nn.Softmax(dim=1)):
    with torch.no_grad():
        output = post_fuction(model(input_tensor))
    return output[0][1].item()


def predict_with_model(
    input_tensor, models, selected_model=None, post_function=nn.Softmax(dim=1)
):
    if selected_model in model_names:
        # If a specific model is selected, only predict with that model
        model_index = model_names.index(selected_model)
        with torch.no_grad():
            output = post_function(models[model_index](input_tensor))
        return [output[0][1].item()]
    else:
        return [0]
