# END-TO-END-DATA-SCIENCE-PROJECT










# 🌸 Iris Flower Classification - FastAPI Deployment Project

This project is an end-to-end machine learning solution that performs Iris flower classification using a pre-trained model deployed via a FastAPI web application. The objective of this internship task was to demonstrate a complete data science workflow — from data preprocessing and model training to deploying the model using a REST API and web interface.

---

## 📌 Project Overview

The Iris dataset is a classic machine learning dataset that consists of 150 records of iris flowers. Each record contains four features:

- Sepal length
- Sepal width
- Petal length
- Petal width

The model predicts one of three flower species:
- Setosa
- Versicolor
- Virginica

We used **scikit-learn** to train a classification model (e.g., Logistic Regression or RandomForest), saved the trained model as a `.pkl` file, and then deployed it using **FastAPI**. The application is hosted on **Render.com** and provides both a Swagger-based API interface and an interactive HTML form for predictions.

---

## 🚀 Features

- 🧠 Trained machine learning model using `train_model.py`
- 🔍 Predict API endpoint (`/predict`) using FastAPI
- 🌐 HTML form frontend for non-technical users (`/`)
- 📊 Interactive API docs via Swagger (`/docs`)
- 🧪 Deployed on Render.com with live prediction capability

---

## 🧱 Tech Stack

- Python 3.10+
- FastAPI
- scikit-learn
- Pandas
- Uvicorn
- Jinja2 (for HTML templates)
- Render (for cloud deployment)

---

## 📁 Project Structure
iris_fastapi_project/
├── app/
│ ├── main.py # FastAPI app with API routes
│ └── model/
│ └── iris_model.pkl # Trained model
├── templates/
│ └── form.html # HTML form for input
├── train_model.py # Script to train the model
├── requirements.txt # All Python dependencies
└── start.sh # Startup script (optional)


---

## ✅ How It Works

1. The model is trained using the Iris dataset and saved as `iris_model.pkl`.
2. FastAPI loads the model and exposes two routes:
   - `/predict`: POST API for predictions using JSON.
   - `/`: GET + POST form for a web-based user input.
3. The web service is hosted via Render. Users can visit the deployed site, enter measurements, and get predictions instantly.

---

## 🌐 Access the App

- 🔗 Live site: [https://your-app-name.onrender.com](https://your-app-name.onrender.com)
- 🔧 Swagger UI: [https://your-app-name.onrender.com/docs](https://your-app-name.onrender.com/docs)

---

## 📌 How to Run Locally

1. Clone the repo:
```bash
git clone https://github.com/yourusername/iris_fastapi_project.git
cd iris_fastapi_project

