
# 🚀 IT Support Ticket Classification & Priority Prediction System  
## 📊 End-to-End Machine Learning Project for Automated Ticket Classification using NLP  

This project builds a complete machine learning pipeline to automatically classify IT support tickets into categories and predict their priority levels using textual data.

The system uses Natural Language Processing (NLP) techniques along with a Logistic Regression model and is deployed using an interactive Streamlit dashboard.

---

## 🌐 Live Application  

🔗 Try the dashboard here:  
 https://futureml02-k5fbgvmmy4m7pmyb2t2ery.streamlit.app/

---

## 📌 Project Overview  

IT support systems generate large volumes of tickets daily. Manually classifying and prioritizing them is inefficient and time-consuming.

This project develops a machine learning-based system that:

- Analyzes ticket descriptions using NLP  
- Classifies tickets into categories  
- Predicts priority levels  
- Provides meaningful outputs for faster issue resolution  

The final system allows users to:

- Classify IT support tickets automatically  
- Predict ticket priority (Low, Medium, High)  
- Understand ticket patterns through visualizations  

---

## 🎯 Problem Statement  

Organizations face challenges such as:

- Manual ticket categorization delays  
- Incorrect prioritization of critical issues  
- Increased response time  
- Reduced operational efficiency  

This project aims to build an intelligent system that automates ticket classification and prioritization using machine learning.

---

## 📂 Dataset Information  

The dataset contains IT support tickets with the following attributes:

- Ticket Description (text data)  
- Category (Hardware, Software, Network, etc.)  
- Priority Level (Low, Medium, High)  

The text data is processed using NLP techniques to extract meaningful features.

---

## 🧠 Machine Learning Models Used  

The system uses:

- Logistic Regression (Primary Model)  

Logistic Regression was selected because:

- Performs well on text classification tasks  
- Works efficiently with TF-IDF features  
- Fast and suitable for real-time prediction  
- Provides interpretable results  

---

## ⚙️ Machine Learning Pipeline  

The project follows a structured ML workflow:

- Data Collection  
- Data Cleaning & Text Preprocessing  
- Feature Engineering (TF-IDF Vectorization)  
- Model Training (Logistic Regression)  
- Model Evaluation  
- Model Selection  
- Prediction System Development  
- Dashboard Deployment  

---

## 📊 Dashboard Features  

The deployed dashboard provides the following functionalities:

---

### 🔴 Ticket Classification System  

Users can:

- Enter a ticket description OR select predefined issues  
- Get predictions for:
  - Category (e.g., Hardware, Software, Network)  
  - Priority Level (Low, Medium, High)  

---

### 📈 Ticket Analytics Dashboard  

Includes visual insights such as:

- Category distribution  
- Priority breakdown  
- Ticket trends  

---

### 📊 Meaningful Prediction Output  

The system provides clear and actionable output:

- 🔍 Predicted Category  
- ⚡ Priority Level  
- 🛠 Suggested Solution  

---

## 📁 Project Structure  

```
Support_Ticket_Classification
│
├── app.py
│
├── src
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── prediction.py
│
├── models
│   ├── category_model.pkl
│   ├── priority_model.pkl
│   └── vectorizer.pkl
│
├── data
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 📊 Model Evaluation  

The model was evaluated using classification metrics:

- Accuracy  
- Precision  
- Recall  
- F1 Score  

Example performance:

| Model                | Accuracy |
|---------------------|----------|
| Logistic Regression | **90% (Selected)** |

Logistic Regression achieved high accuracy with minimal computational cost, making it ideal for real-time ticket classification systems.

---

## 💻 Technologies Used  

### Programming Language  

- Python  

### Libraries  

- Pandas  
- NumPy  
- Scikit-learn  
- NLP (TF-IDF)  
- Matplotlib  
- Seaborn  
- Plotly  

### Framework  

- Streamlit  

### Deployment  

- Streamlit Cloud  

---

## ▶️ How to Run the Project Locally  

Clone the repository  

```
git clone https://github.com/your-username/your-repo-name.git
```

Navigate to the project folder  

```
cd your-repo-name
```

Install dependencies  

```
pip install -r requirements.txt
```

Run the dashboard  

```
streamlit run app.py
```

---

## 📷 Dashboard Preview  

- Ticket Input Interface  
- Prediction Output Panel  
- Analytics Dashboard  
- Graphical Insights  

---

## 📈 Future Improvements  

Possible improvements for the system:

- Integrate deep learning models (BERT / Transformers)  
- Add real-time ticket ingestion (API integration)  
- Multi-language support  
- Automated response suggestion system  
- Deploy using Docker or cloud platforms  

---


