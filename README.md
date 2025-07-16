# Heart Failure Prediction using Machine Learning

## Overview

Cardiovascular diseases (CVDs) remain the **leading cause of death globally**, accounting for nearly **17.9 million deaths annually**, with **heart failure** being one of the most prevalent and deadly outcomes. Early detection and intervention are crucial for at-risk individuals, such as those with hypertension, diabetes, and hyperlipidaemia. This project utilizes the **Heart Failure Prediction dataset**, sourced from Kaggle, to build machine learning models that accurately predict the presence of heart disease.

---

## Project Aim

The goal of this project is to **develop and compare four machine learning algorithms** to classify whether a patient has heart disease (label: 1) or not (label: 0), using a variety of clinical measurements and indicators.

We aim to provide **reliable, data-driven models** that healthcare professionals can potentially integrate into clinical workflows. Each model is fairly trained and evaluated on the same dataset splits to ensure a meaningful comparison.

---

## Objectives

- ✅ Conduct comprehensive **Exploratory Data Analysis (EDA)**.
- ✅ Preprocess and clean the dataset (handle duplicates, encode categorical variables, scale features).
- ✅ Build and evaluate the following machine learning models:
  - **Logistic Regression** using `statsmodels`
  - **Random Forest Classifier** using `scikit-learn`
  - **Support Vector Machine (SVM)**
  - **Decision Tree**
- ✅ Compare model performance using metrics like **accuracy**, **precision**, **recall**, **F1-score**, and **ROC-AUC**.
- ✅ Visualize model results and insights for better interpretability.

---

## Dataset Details

**Source**: [Kaggle – Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

The dataset is a **consolidation of five major heart disease datasets**, cleaned and deduplicated to result in **918 final observations**.

### 🧾 Features

| Feature          | Description                                                                                      |
|------------------|--------------------------------------------------------------------------------------------------|
| `Age`            | Patient's age (years)                                                                            |
| `Sex`            | Biological sex (`M` or `F`)                                                                      |
| `ChestPainType`  | Type of chest pain (`TA`, `ATA`, `NAP`, `ASY`)                                                  |
| `RestingBP`      | Resting blood pressure (mmHg)                                                                    |
| `Cholesterol`    | Serum cholesterol (mg/dl)                                                                        |
| `FastingBS`      | Fasting blood sugar > 120 mg/dl (`1` if true, `0` otherwise)                                     |
| `RestingECG`     | Resting electrocardiogram results (`Normal`, `ST`, `LVH`)                                        |
| `MaxHR`          | Maximum heart rate achieved                                                                      |
| `ExerciseAngina` | Exercise-induced angina (`Y` or `N`)                                                             |
| `Oldpeak`        | ST depression induced by exercise relative to rest                                               |
| `ST_Slope`       | Slope of peak exercise ST segment (`Up`, `Flat`, `Down`)                                        |
| `HeartDisease`   | **Target variable** – `1` indicates heart disease, `0` means normal                              |

---

## Source Breakdown

| Dataset Name         | Observations |
|----------------------|--------------|
| Cleveland            | 303          |
| Hungarian            | 294          |
| Switzerland          | 123          |
| Long Beach VA        | 200          |
| Stalog (Heart) Data  | 270          |
| **Total (after cleaning)** | **918** |

---

## Tools and Libraries

- Python
- `pandas`, `numpy` – data manipulation
- `matplotlib`, `seaborn` – visualizations
- `scikit-learn` – ML models and evaluation
- `statsmodels` – logistic regression
- `joblib` – model saving

---

## Evaluation Metrics

Each model is evaluated based on:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC Curve & AUC Score
- Confusion Matrix

---

## Results

Model performance results will be included here after training and evaluation with visualizations and comparative tables.

---

## Limitations

- Dataset size is relatively small (~918 samples).
- Imbalanced class distribution may affect model fairness.
- External clinical validation is required before real-world application.

---

## Citation

> **fedesoriano.** (September 2021). *Heart Failure Prediction Dataset*. Retrieved from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

---

## Acknowledgements

Thanks to the original data contributors:

- Hungarian Institute of Cardiology, Budapest – *Dr. Andras Janosi*
- University Hospitals in Switzerland – *Dr. William Steinbrunn* and *Dr. Matthias Pfisterer*
- VA Medical Center, Long Beach & Cleveland Clinic – *Dr. Robert Detrano*
- Data compilation by *David W. Aha*

---

## Conclusion

This project not only showcases the power of machine learning in healthcare but also emphasizes the importance of clean data, proper evaluation, and interpretability in building trustworthy diagnostic tools. Future work could involve advanced ensemble models, feature importance interpretation with SHAP, and integration with clinical workflows.
