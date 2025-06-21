# 🧠 Extrovert vs. Introvert Behavior Classification

This project classifies individuals as **Introverts** or **Extroverts** based on behavioral data using Machine Learning models like SVM, MLP, and Linear Regression. It includes both a training pipeline in Jupyter/Colab and a deployed interactive **Streamlit web app**.

---

## 📁 Project Structure

```
Extrovert-vs.-Introvert-Classification/
├── app.py                   # Streamlit app interface
├── Project.ipynb            # Jupyter/Colab notebook for training and analysis
├── models/                  # Serialized trained models (.pkl files)
│   ├── svm_pipeline.pkl
│   ├── linear_pipeline.pkl
│   └── mlp_pipeline.pkl
├── requirements.txt         # Python dependencies
└── README.md                # Documentation
```

---

## 📊 Dataset

* **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data)
* **Attributes**:

  * `Time_spent_Alone`
  * `Stage_fear`
  * `Social_event_attendance`
  * `Going_outside`
  * `Drained_after_socializing`
  * `Friends_circle_size`
  * `Post_frequency`
  * `Personality` (target: Extrovert = 0, Introvert = 1)

---

## 🔍 Features Used

```python
[
    'Time_spent_Alone',
    'Stage_fear',
    'Social_event_attendance',
    'Going_outside',
    'Drained_after_socializing',
    'Friends_circle_size',
    'Post_frequency'
]
```

---

## 🛠️ ML Models Trained

1. **Support Vector Machine (SVM)**
2. **Linear Regression**
3. **Multi-Layer Perceptron (MLP) Classifier**

Each model is wrapped in a **`Pipeline`** with `StandardScaler` for feature normalization.

All models are saved in the `models/` directory using `joblib`.

---

## 📈 Model Evaluation

Each model is evaluated using:

* **Accuracy**
* **Confusion Matrix**
* **Classification Report**

Results (on 80/20 split):

* **SVM**: \~73% Accuracy
* **MLP**: \~75% Accuracy
* **Linear Regression**: Used more for analysis; not ideal for classification

---

## 🎮 Streamlit Web App

A user-friendly Streamlit app is provided for real-time predictions.

### How to Use

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2. **Run the app**:

```bash
streamlit run app.py
```

3. **Interact**:

* Provide your responses to behavioral questions.
* Choose a model (SVM, Linear Regression, MLP).
* Get a prediction: Extrovert or Introvert.

---

## 📦 Sample Input in Streamlit

| Feature                 | Example |
| ----------------------- | ------- |
| Time Spent Alone        | 3       |
| Stage Fear              | No      |
| Social Events/Month     | 5       |
| Going Outside/Week      | 4       |
| Tired After Socializing | Yes     |
| Number of Close Friends | 8       |
| Social Media Posts      | 6       |

---

## 📁 Models Directory

The following pre-trained models are available:

* `svm_pipeline.pkl`
* `linear_pipeline.pkl`
* `mlp_pipeline.pkl`

Each model expects 7 input features in the same order as the dataset.

---

## 🔄 Improvements You Can Make

* Add a **Decision Tree** classifier (code commented out).
* Include **cross-validation** for more reliable performance.
* Optimize hyperparameters (GridSearchCV).
* Add **interactive graphs** in Streamlit (feature importance, confidence, etc.)
