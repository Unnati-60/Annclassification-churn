# 📉 Customer Churn Prediction
## [View Project](https://annclassification-churn-unmr4vpkphxlwrbvscolhc.streamlit.app/)
This project predicts customer churn for a banking dataset using an Artificial Neural Network (ANN). It includes a complete data preprocessing pipeline, model training with TensorFlow, hyperparameter tuning, and an interactive Streamlit web app for real-time predictions.

## Tech Stack
TensorFlow / Keras | Pandas, NumPy | Scikit-learn |Streamlit

## 🧹 Data Preprocessing Pipeline
- Handled categorical features using One-Hot Encoding and Label Encoding
- Scaled numerical features for neural network compatibility
- Ensured consistent feature alignment between training and inference data

## 🧠 Model Development
- Built an ANN model using TensorFlow/Keras
- Tuned key hyperparameters (number of layers, neurons, activation functions, learning rate) using Random Search
- Selected the best-performing architecture based on validation performance

## 📊 Evaluation
- Evaluated model using Recall as the primary metric to better capture churn cases
-Achieved approximately 90% prediction accuracy
