
# MediPredict – Diabetes Prediction System

MediPredict is a Machine Learning-based web application that predicts whether a person is likely to have diabetes based on medical input parameters.

This project was developed as a second-year academic project to explore Python, Machine Learning, and Streamlit.

## Features

- Diabetes prediction using Machine Learning
- Logistic Regression classification model
- Data preprocessing using Pandas
- Feature scaling using StandardScaler
- Interactive web interface using Streamlit
- Prediction result displayed through the application

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

## Project Structure

```text
Medical-Prediction-Project/
│
├── diabetes_app.py
├── diabetes_prediction.py
├── diabetes.csv
├── diabetes_model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

1. Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

2. Open the project folder:

```bash
cd Medical-Prediction-Project
```

3. Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run the application:

```bash
streamlit run diabetes_app.py
```

## Machine Learning Model

The project uses Logistic Regression for binary classification.

The dataset is divided into training and testing sets. StandardScaler is fitted only on training data to avoid data leakage.

Reported test accuracy: **71.43%**

## Disclaimer

This project is developed for educational purposes only. It is not a clinically validated medical diagnostic tool and must not be used for medical decisions.

## Author

Abdul Hadi

GitHub: https://github.com/lonehadi709-png
