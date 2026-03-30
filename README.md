# 🧠 AI Text Classification System

An end-to-end machine learning system for classifying text into 20 different topics using the 20 Newsgroups dataset.
The project includes data preprocessing, model training, evaluation, a RESTful API, and an interactive web interface.

---

## 🚀 Features

* 🔍 Text classification into 20 categories (e.g., sports, politics, technology, science)
* ⚙️ TF-IDF feature extraction
* 🤖 Machine Learning model (SVM) with hyperparameter tuning (GridSearchCV)
* 📊 Model evaluation (accuracy, classification report, confusion matrix)
* 🌐 REST API for real-time prediction
* 🖥️ Interactive UI for user input and visualization
* 🏆 Top-K predictions with confidence scores

---

## 🏗️ System Architecture

```
User Interface (Streamlit)
        ↓
    REST API (FastAPI)
        ↓
 Machine Learning Model (SVM)
```

---

## 📂 Project Structure

```
project/
│
├── api/
│   └── app.py                # FastAPI application
│
├── ui/
│   └── app.py               # Streamlit UI
│
├── src/
│   ├── data_loader.py       # Load dataset
│   ├── preprocess.py        # TF-IDF vectorization
│   ├── model.py             # Training + GridSearch
│   ├── evaluate.py          # Metrics + confusion matrix
│   └── utils.py             # Save/load model
│
├── models/
│   └── svm_model.pkl        # Trained model
│
├── outputs/
│   └── confusion_matrix.png # Evaluation visualization
│
├── main.py                  # Training pipeline
├── predict.py               # Prediction logic
├── requirements.txt
└── README.md
```

---

## 🧠 Model Details

* **Algorithm**: Support Vector Machine (SVM)
* **Feature Extraction**: TF-IDF
* **Hyperparameter Tuning**: GridSearchCV
* **Evaluation Metrics**:

  * Accuracy
  * Precision / Recall / F1-score
  * Confusion Matrix

---

## 📊 Performance

* **Accuracy**: ~84% on test set
* Strong performance across diverse domains:

  * Technology
  * Sports
  * Politics
  * Science

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo

pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Train the model

```bash
python main.py
```

---

### 2. Run API

```bash
uvicorn api.app:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

### 3. Run UI

```bash
streamlit run ui/app.py
```

Open:

```
http://localhost:8501
```

---

## 🧪 Example

Input:

```
NASA launched a rocket into space
```

Output:

```json
{
  "label": "sci.space",
  "confidence": 0.99
}
```

---

## 📸 Demo

> Add screenshots of:
>
> * UI interface
> * API response
> * Confusion matrix

---

## 🛠️ Technologies Used

* Python
* scikit-learn
* FastAPI
* Streamlit
* Matplotlib

---

## 💡 Future Improvements

* 🔧 Add deep learning models (BERT, LSTM)
* 🌍 Deploy to cloud (AWS / Render / Docker)
* 📈 Model comparison (Logistic Regression vs SVM)
* 🧾 Logging and monitoring system

---

## 📌 Key Learning Outcomes

* Built a complete machine learning pipeline
* Designed and deployed a REST API for ML inference
* Developed an interactive UI for real-time predictions
* Learned how to structure production-ready ML projects

---

## 👨‍💻 Author
Nam Nguyen Thai Hoai
---

## ⭐ If you find this project useful, give it a star!
