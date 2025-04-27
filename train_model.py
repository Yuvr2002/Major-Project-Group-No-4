import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Directories for the dataset
defective_dir = 'dataset/defective'
non_defective_dir = 'dataset/non_defective'

# Load images and labels
def load_data():
    images = []
    labels = []

    for label, dir_path in enumerate([defective_dir, non_defective_dir]):
        for image_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, image_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (64, 64)).flatten() / 255.0
            images.append(img)
            labels.append(label)
    
    return np.array(images), np.array(labels)

# Prepare data
X, y = load_data()

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the model
model_path = 'backend/model/model.pkl'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")