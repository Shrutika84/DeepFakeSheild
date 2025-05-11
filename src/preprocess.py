# src/preprocess.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

IMG_SIZE = 128


def load_images_from_folder(folder_path, label):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img / 255.0)  # normalize to [0,1]
                labels.append(label)
    return images, labels


def load_dataset(real_dir, fake_dir):
    real_images, real_labels = load_images_from_folder(real_dir, 0)
    fake_images, fake_labels = load_images_from_folder(fake_dir, 1)

    X = np.array(real_images + fake_images)
    y = np.array(real_labels + fake_labels)

    return train_test_split(X, y, test_size=0.2, random_state=42)


if __name__ == "__main__":
    real_path = "../Dataset/real"
    fake_path = "../Dataset/fake"

    X_train, X_test, y_train, y_test = load_dataset(real_path, fake_path)
    print("Dataset loaded:")
    print("Train samples:", len(X_train))
    print("Test samples:", len(X_test))
