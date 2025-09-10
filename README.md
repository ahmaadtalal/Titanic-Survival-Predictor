# Titanic Survival Prediction

## Project Overview
**Objective:** Predict whether a passenger survived the Titanic disaster using passenger information.  
**Type:** Binary classification (0 = Did Not Survive, 1 = Survived)  
**Dataset:** Titanic dataset (commonly from Kaggle Titanic Competition)  

---

## Data Exploration (EDA)
- Checked missing values and summary statistics.  
- **Key observations:**  
  - `Age` had missing values → filled with median.  
  - `Embarked` had missing values → filled with mode.  
  - `Cabin` dropped due to too many missing entries.  
- **Visualizations:**  
  - Survival by Gender → Females survived more.  
  - Survival by Age → Children had higher survival.  
  - Survival by Pclass → Higher class had higher survival rate.  
  - Survival by Embarked → Certain ports had higher survival.  

---

## Data Preprocessing
- **Handling Missing Values:**  
  - `Age` → median  
  - `Embarked` → mode  
  - `Cabin` → dropped  

- **Encoding Categorical Variables:**  
  - `Sex` → 0 (male), 1 (female)  
  - `Embarked` → One-hot: `Embarked_C`, `Embarked_Q`, `Embarked_S` (all retained)  

- **Feature Engineering:**  
  - `FamilySize` = `SibSp + Parch + 1`  
  - `IsAlone` = 1 if `FamilySize` = 1 else 0  
  - `Title` extraction from Name (`Mr`, `Miss`, `Mrs`, `Master`, `Rare`)  

- **Final Features (16 columns):**  
  `Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_C, Embarked_Q, Embarked_S, FamilySize, IsAlone, Title_Master, Title_Miss, Title_Mr, Title_Mrs, Title_Rare`  

---

## Model Building & Evaluation
- **Train/Test Split:** 80/20  

- **Models Trained:**  
  - Logistic Regression ✅  
  - Random Forest ✅  
  - Support Vector Machine (SVM) ✅  
  - Gradient Boosting ✅  
  - XGBoost ✅  

- **Evaluation Metrics:**  
  - Accuracy, Precision, Recall, F1-score  
  - Confusion Matrix  
  - Cross-validation  

- **Best Observations:**  
  | Model                | Accuracy |
  |---------------------|----------|
  | Logistic Regression  | 0.85     |
  | Random Forest        | 0.83     |
  | Gradient Boosting    | 0.83     |
  | XGBoost              | 0.82     |
  | SVM                  | 0.63     |

- **Feature Importance (Gradient Boosting & Random Forest):**  
  - Most important features: `Title_Mr`, `Fare`, `Pclass`, `Age`, `FamilySize`, `Sex`  

- **Hyperparameter Tuning:**  
  - Random Forest → `n_estimators`, `max_depth`, `min_samples_leaf/split`  
  - Gradient Boosting → `learning_rate`, `n_estimators`, `max_depth`  

---

## Model Deployment (Local Testing)
- Saved best model (Logistic Regression) using `pickle`.  

- **Sample Prediction:**
```python
sample_input = {
    "Pclass": 3, "Sex": 0, "Age": 22, "SibSp": 1, "Parch": 0, "Fare": 7.25,
    "Embarked_C": 0, "Embarked_Q": 0, "Embarked_S": 1,
    "FamilySize": 2, "IsAlone": 0,
    "Title_Master": 0, "Title_Miss": 0, "Title_Mr": 1, "Title_Mrs": 0, "Title_Rare": 0
}

Prediction: 0  
Probability: 0.08  
Result: NOT SURVIVE
