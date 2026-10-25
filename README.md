# Titanic Survival Prediction API 🚢

## 1. Project Overview

This project develops a Machine Learning model to predict whether a Titanic passenger would survive based on passenger information such as age, gender, passenger class, fare, and family details.

The trained model is deployed using **FastAPI**, allowing users to submit passenger information through a REST API and receive a survival prediction with an estimated probability.

## 2. Business Problem

The objective is to analyze passenger characteristics and build a classification model that predicts survival outcomes.

This project demonstrates how Machine Learning can transform historical data into predictions and how a trained model can be exposed through an API for application integration.

**Problem type:** Binary Classification

* `0` — Did not survive
* `1` — Survived

## 3. Technologies Used

* **Python** — Programming and data analysis
* **Pandas** — Data manipulation
* **Scikit-learn** — Preprocessing, model training, and evaluation
* **Logistic Regression** — Primary classification model
* **Random Forest** — Model comparison
* **Joblib** — Model serialization and loading
* **FastAPI** — REST API development
* **Uvicorn** — API server
* **Jupyter Notebook** — Experimentation and analysis
* **Swagger UI** — Interactive API testing
* **Git and GitHub** — Version control and project hosting

## 4. Machine Learning Workflow

The project follows an end-to-end Machine Learning workflow:

1. Load the Titanic dataset.
2. Inspect the dataset structure and data types.
3. Analyze missing values and duplicate records.
4. Perform Exploratory Data Analysis (EDA).
5. Remove columns that could cause target leakage.
6. Prepare features and the target variable.
7. Split the data into training and testing sets.
8. Handle missing values and encode categorical features.
9. Scale numerical features.
10. Train Logistic Regression and Random Forest models.
11. Evaluate models using classification metrics and ROC-AUC.
12. Perform cross-validation and hyperparameter tuning.
13. Save the selected trained model.
14. Build a REST API using FastAPI.
15. Test predictions using Swagger UI.

## 5. Dataset and Features

The dataset contains passenger information, including the following features:

| Feature    | Description                          |
| ---------- | ------------------------------------ |
| `pclass`   | Passenger class: 1, 2, or 3          |
| `sex`      | Passenger gender                     |
| `age`      | Passenger age                        |
| `sibsp`    | Number of siblings or spouses aboard |
| `parch`    | Number of parents or children aboard |
| `fare`     | Passenger fare                       |
| `embarked` | Port of embarkation                  |
| `survived` | Target variable: 0 or 1              |

The selected model uses seven input features: `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, and `embarked`.

The target variable is `survived`.

The columns `boat` and `body` were removed to avoid target leakage because they can reveal information about the passenger's survival outcome.

## 6. Data Preprocessing

The following preprocessing steps were implemented:

* Missing numerical values were handled using median imputation.
* Missing categorical values were handled using most-frequent imputation.
* Categorical variables were encoded using One-Hot Encoding.
* Numerical features were standardized using StandardScaler.
* The dataset was split into training and testing sets using an 80:20 ratio.
* Stratified splitting was used to preserve the target class distribution.

A Scikit-learn Pipeline and ColumnTransformer were used to keep preprocessing and prediction consistent.

## 7. Model Training and Evaluation

Two classification algorithms were compared:

* Logistic Regression
* Random Forest Classifier

### Logistic Regression Results

| Metric             | Result |
| ------------------ | -----: |
| Test Accuracy      | 80.92% |
| ROC-AUC            | 0.8670 |
| Survivor Precision |   0.77 |
| Survivor Recall    |   0.71 |
| Survivor F1-Score  |   0.74 |

### Random Forest Results

| Metric             | Result |
| ------------------ | -----: |
| Test Accuracy      | 79.01% |
| ROC-AUC            | 0.8636 |
| Survivor Precision |   0.71 |
| Survivor Recall    |   0.77 |
| Survivor F1-Score  |   0.74 |

### Model Selection

Logistic Regression was selected as the final model because it achieved slightly higher test accuracy and ROC-AUC in this experiment.

Random Forest achieved higher recall for the survivor class, making it a useful comparison when identifying more actual survivors is a priority.

Five-fold cross-validation and GridSearchCV were used to assess model stability and tune Logistic Regression hyperparameters.

The best cross-validation accuracy was approximately 78.70%.

*These metrics represent results from this experiment and may vary if the dataset or experimental setup changes.*

## 8. Project Structure

```text
titanic-survival-prediction-api/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── titanic_survival_model.pkl
│
├── notebooks/
│   └── titanic_survival_analysis.ipynb
│
└── screenshots/
    └── api_prediction.png
```

## 9. Installation and Setup

### Prerequisites

* Python 3.10 or a compatible version supported by the required libraries
* pip
* Git (optional)

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/titanic-survival-prediction-api.git
```

Replace `YOUR_USERNAME` with your GitHub username.

### Step 2: Navigate to the Project

```bash
cd titanic-survival-prediction-api
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Start the API

```bash
python -m uvicorn main:app --reload
```

The API should now be available locally.

## 10. API Endpoints

| Method | Endpoint   | Purpose                                |
| ------ | ---------- | -------------------------------------- |
| GET    | `/`        | Check whether the API is running       |
| POST   | `/predict` | Predict passenger survival             |
| GET    | `/docs`    | Open interactive Swagger documentation |

### Home Endpoint

URL:

`http://127.0.0.1:8000/`

Example response:

```json
{
  "message": "Titanic Survival Prediction API is running!"
}
```

## 11. Make a Prediction

Open Swagger UI:

`http://127.0.0.1:8000/docs`

Select `POST /predict`, click **Try it out**, and submit the following JSON:

```json
{
  "pclass": 1,
  "sex": "female",
  "age": 25,
  "sibsp": 0,
  "parch": 0,
  "fare": 80.0,
  "embarked": "C"
}
```

### Example Response

```json
{
  "prediction": 1,
  "survival_status": "Likely to survive",
  "survival_probability": 0.9556
}
```

The example response illustrates the response format. The actual prediction and probability depend on the saved model.

* `prediction`: Predicted class, either 0 or 1.
* `survival_status`: Human-readable prediction.
* `survival_probability`: Estimated probability of survival, expressed as a decimal between 0 and 1.

The prediction is an estimate, not a guarantee of an actual outcome.

## 12. Key Learnings

Through this project, I practiced:

* Exploratory Data Analysis and data quality checks.
* Data cleaning and missing-value treatment.
* Feature engineering and categorical encoding.
* Training and comparing classification algorithms.
* Evaluating models using precision, recall, F1-score, accuracy, and ROC-AUC.
* Cross-validation and hyperparameter tuning.
* Saving and loading trained Machine Learning pipelines.
* Developing and testing REST APIs with FastAPI.
* Using Swagger UI for interactive API testing.
* Organizing a Machine Learning project for GitHub.

## 13. Future Improvements

Potential improvements include:

* Add automated unit and API tests.
* Add structured logging and better exception handling.
* Containerize the application using Docker.
* Deploy the API to a cloud hosting platform.
* Add continuous integration with GitHub Actions.
* Monitor API performance and prediction behavior.

## 14. Disclaimer

This project is intended for educational and portfolio purposes. The predictions are based on historical data and should not be interpreted as causal conclusions or guarantees.

## 15. Author

**Saswat Betta Aptakam**

GitHub: `https://github.com/saswat925`

---

⭐ If you find this project useful, consider starring the repository.
