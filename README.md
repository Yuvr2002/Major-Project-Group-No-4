Shield Drug AI - Tablet Defect Detection
Project Overview
Shield Drug AI is a web-based application that helps detect whether a pharmaceutical tablet is Defective or Non-Defective. Users can log in, upload tablet images, and receive instant predictions powered by Machine Learning and Computer Vision.

Features
User Login and Authentication

Upload tablet images

Predict defective or non-defective tablets

Display uploaded images alongside results

Responsive and simple web interface

Technologies Used
Backend: Python, Flask, Machine Learning (Random Forest Classifier)

Frontend: HTML, CSS (Bootstrap), JavaScript

Libraries: OpenCV, NumPy, Scikit-learn, Joblib

Tools: Visual Studio Code (VS Code)

Project Structure
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
How to Run Locally
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/Shield-Drug-AI.git
Install required packages:

nginx
Copy
Edit
pip install -r requirements.txt
Train the model if not already done:

nginx
Copy
Edit
python train_model.py
Start the Flask server:

bash
Copy
Edit
cd backend
python app.py
Open your browser and go to http://127.0.0.1:5000/.

Expected Outcome
Users can log in, upload tablet images, and accurately predict whether the tablet is defective or not, with the result and image displayed clearly on the result page.

