**Shield Drug AI - Tablet Defect Detection System**
**Introduction**
Shield Drug AI is a simple yet powerful web-based system designed to detect whether pharmaceutical tablets are Defective or Non-Defective. By using Machine Learning and Computer Vision techniques, the system ensures quick and accurate quality control for pharmaceutical products.

**Objective**
The main goal of the project is to assist pharmaceutical industries by providing an automated way to verify the quality of tablets based on their visual features. This reduces human errors and improves product safety.

**Features**
**User Authentication**: Login system to allow users to securely access the platform.

**Tablet Image Upload**: Users can upload tablet images through the web interface.

**Defect Detection**: The system predicts whether the uploaded tablet is defective or non-defective.

**Result Display**: Uploaded images and prediction results are displayed together.

**Simple Interface**: Clean and easy-to-use web interface for smooth user experience.

**Technologies Used**
**Frontend**: HTML, CSS (Bootstrap), JavaScript

**Backend**: Python, Flask Framework

**Machine Learning**: Random Forest Classifier (trained on tablet images)

**Libraries**: OpenCV, NumPy, Scikit-learn, Joblib

**Tool**: Visual Studio Code (VS Code)

**Project Structure**
pgsql
Copy
Edit
tablet_defect_detection/
│── backend/
│   ├── app.py
│   ├── model/
│   │   ├── model.pkl
│   ├── static/
│   │   ├── style.css
│   ├── templates/
│   │   ├── login.html
│   │   ├── index.html
│   │   ├── result.html
│   ├── uploads/
│── dataset/
│   ├── defective/
│   ├── non_defective/
│── train_model.py
│── requirements.txt
│── README.md

**How It Works**
The user logs into the system.

The user uploads an image of a tablet.

The backend model processes the image and predicts its quality.

The system displays whether the tablet is "Defective" or "Non-Defective" along with the uploaded image.

**Industry Use Case**
This system can be used in the pharmaceutical industry for automated tablet inspection, ensuring that defective tablets are detected early and do not reach customers, thereby improving overall product quality and safety.

**Expected Outcome**
An easy-to-use web application where users can upload images of tablets and instantly receive defect predictions, helping maintain high standards in pharmaceutical production.
