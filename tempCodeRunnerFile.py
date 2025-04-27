import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_data(data_dir):
    images, labels = [], []
    for label in ['defective', 'non_defective']:
        label_dir = os.path.join(data_dir, label)
        if not os.path.exists(label_dir):
            print(f"Warning: {label_dir} does not exist!")
            continue

        for filename in os.listdir(label_dir):
            img_path = os.path.join(label_dir, filename)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not read {img_path}, skipping.")
                continue

            image = cv2.resize(image, (64, 64)).flatten() / 255.0  # Normalize
            images.append(image)
            labels.append(1 if label == 'defective' else 0)

    return np.array(images), np.array(labels)

def train_model():
    data_dir = r"D:\tablets_quality_test\dataset"
    images, labels = load_data(data_dir)

    if len(images) == 0:
        raise ValueError("No images found! Check your dataset.")

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save trained model
    os.makedirs("backend/model", exist_ok=True)
    joblib.dump(model, "backend/model/model.pkl")

    print("✅ Model trained and saved as 'backend/model/model.pkl'")

if __name__ == '__main__':
    train_model()
