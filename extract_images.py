import os
import cv2
import random


def extract_frames(input_folder, output_folder, num_frames=10, scale_factor=1.3):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".mp4"):
                video_path = os.path.join(root, file)
                output_path = os.path.join(
                    output_folder, os.path.relpath(video_path, input_folder)
                )
                os.makedirs(output_path, exist_ok=True)

                cap = cv2.VideoCapture(video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                for i in range(num_frames):
                    rand_idx = random.randint(0, frame_count - 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, rand_idx)
                    ret, frame = cap.read()

                    if ret:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(
                            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                        )

                        if len(faces) > 0:
                            x, y, w, h = faces[0]
                            # Calculate the new dimensions for cropping
                            new_w = int(w * scale_factor)
                            new_h = int(h * scale_factor)
                            # Calculate the new coordinates for cropping
                            new_x = max(x - int((new_w - w) / 2), 0)
                            new_y = max(y - int((new_h - h) / 2), 0)
                            new_x_end = min(new_x + new_w, frame.shape[1])
                            new_y_end = min(new_y + new_h, frame.shape[0])
                            cropped_frame = frame[new_y:new_y_end, new_x:new_x_end]

                            # Determine the appropriate subfolder for saving
                            subfolder = (
                                "original" if "original" in output_path else "altered"
                            )
                            frame_filename = os.path.join(
                                output_folder, subfolder, f"frame_{i}.jpg"
                            )
                            cv2.imwrite(frame_filename, cropped_frame)
                            print(f"Saved cropped frame {i} from {file}")

                cap.release()


if __name__ == "__main__":
    root_dir = "/Users/nick/deepfake_compressed/"
    output_dir = "/Users/nick/lol/"

    train_original_dir = os.path.join(root_dir, "train", "original")
    train_altered_dir = os.path.join(root_dir, "train", "altered")
    val_dir = os.path.join(root_dir, "val")
    test_dir = os.path.join(root_dir, "test")

    os.makedirs(os.path.join(output_dir, "train", "original"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train", "altered"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

    extract_frames(train_original_dir, output_dir)
    extract_frames(train_altered_dir, output_dir)
    extract_frames(val_dir, output_dir)
    extract_frames(test_dir, output_dir)
