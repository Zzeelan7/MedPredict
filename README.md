# MedPredict
A real-time data science project that uses machine learning to analyze and predict student performance based on academic records, attendance, quiz scores, and engagement metrics. The goal is to provide actionable insights for early intervention and personalized learning strategies.

# 🧑‍⚕️ Disease Prediction System (Diabetes, Heart Disease, Parkinson’s)

This project uses **Machine Learning** to predict whether a person is at risk of:  
- **Diabetes**  
- **Heart Disease**  
- **Parkinson’s Disease**  

based on their **medical attributes and test values**.  

The goal is to provide an easy-to-use tool for **early detection and awareness**, supporting both individuals and healthcare professionals.

---

## 🚀 Features
- Predicts risk of **Diabetes, Heart Disease, and Parkinson’s**  
- Interactive **Streamlit UI** for input and predictions  
- Models saved as **`.pkl` files** for reuse without retraining  
- Simple & extendable (add more diseases/models easily)  

---

## 📂 Dataset
The project is trained on 3 standard medical datasets (publicly available):  
- **Diabetes Dataset** (Pima Indians Diabetes Database)  
- **Heart Disease Dataset** (Cleveland Heart Disease Database)  
- **Parkinson’s Dataset** (UCI Parkinson’s Disease Dataset)  

Each dataset contains **medical features** such as:  
- Diabetes → glucose level, BMI, age, etc.  
- Heart Disease → cholesterol, blood pressure, chest pain type, etc.  
- Parkinson’s → vocal frequency measures, jitter, shimmer, etc.  

---

## 🛠️ Tech Stack
- **Python**  
- **Pandas, NumPy** – Data preprocessing  
- **Scikit-learn** – Machine Learning (classification models)  
- **Streamlit** – Interactive Web UI  

---

## 📊 Model
- Models trained separately for each disease  
- Saved as:  
  - `diabetes.pkl`  
  - `heart_disease.pkl`  
  - `parkinson.pkl`  
- Classification algorithms (e.g., Random Forest, SVM, Logistic Regression) were used  
- Evaluation metrics: **Accuracy, Precision, Recall**  

---

🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to open a Pull Request.
