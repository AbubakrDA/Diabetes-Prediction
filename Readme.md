# Diabetes Prediction Project

A machine learning project that predicts the likelihood of diabetes in patients using various health metrics. Built with scikit-learn and deployed with FastAPI and Docker.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Local Development](#local-development)
  - [Docker Deployment](#docker-deployment)
  - [API Endpoints](#api-endpoints)
- [Model Details](#model-details)
- [Project Structure](#project-structure)
- [CI/CD Pipeline](#cicd-pipeline)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a diabetes prediction system using machine learning. It compares multiple classification algorithms (Logistic Regression, KNN, SVC, Decision Tree, Random Forest, Gradient Boosting) to identify the best performing model for diabetes prediction.

## ✨ Features

- **Multiple ML Models**: Compares 6 different classification algorithms
- **Data Preprocessing**: Handles missing values and data normalization
- **FastAPI REST API**: Easy-to-use API for predictions
- **Docker Support**: Containerized application for easy deployment
- **Batch Predictions**: Predict for multiple patients at once
- **Health Checks**: Built-in API health monitoring
- **CI/CD Pipeline**: GitHub Actions workflow for automated testing and deployment
- **Interactive Documentation**: Swagger UI and ReDoc documentation

## 📊 Dataset

The project uses the Pima Indians Diabetes Database containing:

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-hour serum insulin (mu U/ml)
- **BMI**: Body mass index
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age in years
- **Outcome**: Diabetes (1) or Not (0)

**Data File**: `diabetesdataset.csv` (768 samples)

## 🚀 Installation

### Prerequisites

- Python 3.9+
- pip or conda
- Docker (optional, for containerized deployment)

### Local Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd "Diabetes Prediction"
```

2. **Create a virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train the model (if needed)**
```bash
jupyter notebook DiabetesPrediction.ipynb
# Run all cells to train and save the model
```

## 💻 Usage

### Local Development

1. **Run the Jupyter Notebook**
```bash
jupyter notebook DiabetesPrediction.ipynb
```

2. **Start the FastAPI server**
```bash
python DiabetesPredictionFastApi.py
```

The API will be available at `http://localhost:8000`

### Docker Deployment

1. **Build the Docker image**
```bash
docker build -t diabetes-prediction:latest .
```

2. **Run the container**
```bash
docker run -p 8000:8000 diabetes-prediction:latest
```

3. **Access the API**
```
http://localhost:8000
```

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 28,
    "Insulin": 100,
    "BMI": 30.5,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 45
  }'
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.85,
  "message": "High risk of diabetes. Please consult a healthcare professional."
}
```

#### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict_batch" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "Pregnancies": 2,
      "Glucose": 120,
      "BloodPressure": 70,
      "SkinThickness": 28,
      "Insulin": 100,
      "BMI": 30.5,
      "DiabetesPedigreeFunction": 0.5,
      "Age": 45
    },
    {
      "Pregnancies": 1,
      "Glucose": 85,
      "BloodPressure": 66,
      "SkinThickness": 29,
      "Insulin": 0,
      "BMI": 26.6,
      "DiabetesPedigreeFunction": 0.351,
      "Age": 31
    }
  ]'
```

#### Interactive API Documentation

Access the interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📈 Model Details

### Models Trained

| Model | Test Accuracy |
|-------|---------------|
| Logistic Regression (LR) | 76.6% |
| K-Nearest Neighbors (KNN) | 68.8% |
| Support Vector Machine (SVC) | 73.4% |
| Decision Tree (DT) | 73.4% |
| Random Forest (RF) | 72.7% |
| Gradient Boosting (GBC) | 75.3% |

**Best Model**: Random Forest with max_depth=3

### Data Preprocessing

1. **Missing Value Handling**: Replaced 0 values with column mean
2. **Feature Scaling**: StandardScaler for KNN and SVC models
3. **Train-Test Split**: 80-20 split with random_state=42

## 📁 Project Structure

```
Diabetes Prediction/
├── DiabetesPrediction.ipynb          # Model training notebook
├── DiabetesPredictionFastApi.py      # FastAPI application
├── diabetesdataset.csv               # Training dataset
├── DiabetesPredictionModel/          # Trained model storage
│   └── diabetes_model.pkl
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Docker configuration
├── Readme.md                         # This file
├── github.yml                        # GitHub Actions CI/CD
└── .github/
    └── workflows/
        └── ci-cd.yml                 # CI/CD workflow
```

## 🔄 CI/CD Pipeline

The project includes a GitHub Actions workflow that:

1. **Tests** - Runs on Python 3.9, 3.10, 3.11
2. **Code Quality** - Pylint and Bandit security checks
3. **Build** - Creates Docker image
4. **Push** - Pushes to Container Registry
5. **Deploy** - Deploys to dev/prod environments

### Workflow Triggers

- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or feedback about this project, please open an issue on GitHub.

## 🙏 Acknowledgments

- Pima Indians Diabetes Database from UCI Machine Learning Repository
- scikit-learn for ML algorithms
- FastAPI for the REST API framework
- Docker for containerization

## 🔗 References

- [scikit-learn Documentation](https://scikit-learn.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
- [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

---

**Last Updated**: January 2026
