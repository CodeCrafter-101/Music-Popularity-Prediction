# Music Popularity Prediction

Predict how popular a song will be based on its audio features using machine learning models: **Random Forest**, **XGBoost**, and **LightGBM**. This project also uses **SHAP** for model explainability.

---

## Problem Statement

Spotify provides various features for each track (e.g., energy, valence, danceability). Our goal is to **predict the song's popularity score (0–100)** using these audio-based metrics.

---

## Dataset

- **Source:** [Kaggle - Spotify Dataset](https://www.kaggle.com/datasets/maharshipandya/spotify-dataset)
- **Features used:**

```text
'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 
'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence'
````

---

## Tech Stack & Tools
- Python
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn, XGBoost, LightGBM, SHAP
- Streamlit (optional) for web deployment 

## Models
| Model         | R² Score |
|---------------|----------|
| Random Forest | 0.78     |
| XGBoost       | 0.81     |
| LightGBM      | 0.80     |

## SHAP Output (Feature Importance)
Top features impacting song popularity:
- Energy
- Danceability
- Valence
- Speechiness

  
## How to Run This Project
- Clone the Repo
  - git clone https://github.com/CodeCrafter-101/music-popularity-prediction.git cd Music-Popularity-Predictio
- Install Requirements
  - ````
    !pip install xgboost lightgbm shap scikit-learn pandas matplotlib seaborn
    ````

---

## Project Structure
![image](https://github.com/user-attachments/assets/7dea625b-1556-4139-8980-8adee05fb4cb)

---

## Libraries required
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- shap
- matplotlib
- seaborn
- joblib

---

## SHAP Explainability
- We used SHAP (SHapley Additive exPlanations) to understand feature importance in predictions.
- Top Features:
  - Energy
  - Danceability
  - Valence
  - Speechiness
  - Loudness
  - SHAP plot example:
    - ````
      import shap
      explainer = shap.Explainer(model, X_train)
      shap_values = explainer(X_test)
      shap.plots.beeswarm(shap_values)
      ````

---

## Model Saving & Reuse
- ````
  import joblib
  joblib.dump(rf_model, 'rf_model.pkl')
  joblib.dump(xgb_model, 'xgb_model.pkl')
  joblib.dump(lgb_model, 'lgb_model.pkl')
  ````
