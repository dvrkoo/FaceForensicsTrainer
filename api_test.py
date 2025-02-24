import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from playerModules.mantranet import pre_trained_model, check_forgery
import torch
from PIL import Image
import os
import numpy as np
import queue
import threading
import cv2
from playerModules import model_functions
import json

folder_to_watch = os.path.abspath("/Users/niccolomarini/Desktop/test/")

output_folder = os.path.join(folder_to_watch, "output")

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
MantraNetmodel = pre_trained_model(
    weight_path="trained_models/MantraNetv4.pt", device=device
)
models_index = ["faceswap", "deepfake", "neuraltextures", "face2face", "faceshifter"]


def update_label(face, predictions, frame):
    predictions = [item for sublist in predictions for item in sublist]
    if predictions == []:
        predictions = [0, 0, 0, 0, 0]
    x, y, width, height = face.left(), face.top(), face.width(), face.height()
    max_index = predictions.index(max(predictions))
    is_fake = 1 if max(predictions) > 0.5 else 0
    label = f"Faked with model: {models_index[max_index]}" if is_fake else "Genuine"
    label_color = (255, 0, 0) if is_fake else (0, 255, 0)

    cv2.putText(
        frame,
        label,
        (x, max(0, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        label_color,
        2,
    )

    cv2.rectangle(
        frame,
        (max(0, x), max(0, y)),
        (min(frame.shape[1], x + width), min(frame.shape[0], y + height)),
        label_color,
        2,
    )


def check_image_mantra(img_path):
    image = Image.open(img_path)

    # Convert to RGB if it's RGBA or grayscale
    if image.mode != "RGB":
        image = image.convert("RGB")

    figs = check_forgery(MantraNetmodel, img_path=img_path, device=device)
    return figs


def process_image(file_path):
    print("Image detected")
    figs = check_image_mantra(file_path)
    print("Figures Processed")
    print(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    base_name = os.path.basename(file_path)
    name_without_ext, _ = os.path.splitext(base_name)
    for key, img in figs.items():
        output_file_name = f"{name_without_ext}_{key}.png"
        output_file_path = os.path.join(output_folder, output_file_name)

        try:
            # If the image is a NumPy array, handle conversion.
            if isinstance(img, np.ndarray):
                # If the array is of a floating point type, normalize to [0, 255]
                if np.issubdtype(img.dtype, np.floating):
                    # Avoid division by zero if the image is constant
                    if np.max(img) != np.min(img):
                        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
                    else:
                        img = np.zeros_like(img)
                    img = img.astype(np.uint8)

                pil_img = Image.fromarray(img)
            else:
                pil_img = img

            # If the PIL image is in mode "F", convert it to "L" (grayscale)
            if pil_img.mode == "F":
                pil_img = pil_img.convert("L")

            pil_img.save(output_file_path)
            print(f"Saved {key} image to: {output_file_path}")
        except Exception as e:
            print(f"Error saving {key} image: {e}")


def predict_frame_from_video(models, input_tensor, predictions, json):
    for model in models_index:
        prediction = model_functions.predict_with_model(
            input_tensor,
            models,
            selected_model=model,
        )
        predictions.append(prediction)
        json[model].append(prediction)


def process_video(file_path):
    print("Video detected")
    pred_json = {}
    for model in models_index:
        pred_json[model] = []
    base_name = os.path.basename(file_path)
    output_file_path = os.path.join(output_folder, base_name)

    cap = cv2.VideoCapture(file_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(output_file_path, fourcc, fps, (width, height))
    models, detector = model_functions.load_models()
    for _ in range(int(total_frames)):
        predictions = []
        ret, frame = cap.read()
        frame = model_functions.convert_color_space(frame)
        faces = model_functions.detect_faces(frame, detector)
        if faces:
            face = faces[0]
            x, y, size = model_functions.get_boundingbox(face, frame)
            face_roi = frame[y : y + size, x : x + size]
            input_tensor = model_functions.preprocess_input(face_roi)
            predict_frame_from_video(models, input_tensor, predictions, pred_json)
            update_label(face, predictions, frame)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
    video_writer.release()
    json_file_path = os.path.splitext(output_file_path)[0] + ".json"
    with open(json_file_path, "w") as json_file:
        json.dump(
            pred_json, json_file, indent=4
        )  # 'indent=4' makes the file easier to read


# Define the function to process the file
def process_file(file_path):
    # Replace this with the actual logic you need.
    # For example, you might read the file and perform some operations.

    print(f"Processing file: {file_path}")
    if file_path.endswith(".jpg") or file_path.endswith(".png"):
        process_image(file_path)
    elif file_path.endswith("mp4"):
        process_video(file_path)
    else:
        print("The file is not an image. Skipping.")

    # clear cuda cache
    torch.mps.empty_cache()


file_queue = queue.Queue()


class FileCreatedHandler(FileSystemEventHandler):
    """Watchdog event handler that enqueues newly created files."""

    def on_created(self, event):
        if not event.is_directory:
            print(f"New file detected and queued: {event.src_path}")
            file_queue.put(event.src_path)


def worker():
    """
    Worker thread that continuously processes files from the queue.
    Files are processed one at a time in the order they were added.
    """
    while True:
        file_path = file_queue.get()  # This will block until an item is available.
        try:
            process_file(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        finally:
            file_queue.task_done()


def main():
    # Folder to monitor (change this path as needed)
    print(f"Monitoring folder: {folder_to_watch}")

    event_handler = FileCreatedHandler()
    observer = Observer()
    observer.schedule(event_handler, path=folder_to_watch, recursive=False)
    observer.start()

    # Start the worker thread as a daemon thread.
    threading.Thread(target=worker, daemon=True).start()

    try:
        # Keep the main thread alive.
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping observer...")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
