## 📩 SMS Spam Detection using Machine Learning

A complete end-to-end machine learning pipeline to classify SMS messages as Spam or Ham (Not Spam) using NLP techniques and traditional ML algorithms.

------------------------------------------------------------------------

## 🚀 Project Overview

This project builds a robust SMS Spam Detection System using Python.
It includes:

-   Data preprocessing & cleaning
-   Exploratory Data Analysis (EDA)
-   Natural Language Processing (NLP)
-   Model building using CountVectorizer and TF-IDF
-   Training multiple ML algorithms
-   Performance evaluation & comparison
-   Deployment-ready pipeline

------------------------------------------------------------------------

## 📂 Dataset

Using the standard Spam SMS Dataset (spam.csv) with two main columns:
```
  Column   Description             
  ------   ----------------------- 
  `v1`     Label (`ham` or `spam`) 
  `v2`     Message text            

```
Renamed as:
-  label
-  message

------------------------------------------------------------------------

## 🧹 Data Preprocessing
✔ Cleaning Steps
-  Lowercasing
-  Removing numbers & punctuation
-  Tokenization
-  Removing stopwords
-  Lemmatization
-   converting text to numeric features using:
      -    Bag of Words (CountVectorizer)
      -    TF-IDF Vectorizer
------------------------------------------------------------------------

## 🧰 Libraries Used
   -   numpy
   -   pandas
   -   matplotlib
   -   seaborn
   -   nltk
   -   scikit-learn
------------------------------------------------------------------------

## 🧠 Machine Learning Models Used

You trained and evaluated multiple algorithms:
  -  Logistic Regression
  -  Multinomial Naïve Bayes
  -  Support Vector Machine (SVM)
  -  Decision Tree
  -  Random Forest
  -  K-Nearest Neighbors (KNN)
    
Each was tested with:
  -  CountVectorizer features
  -  TF-IDF features

------------------------------------------------------------------------

## 📊 Model Evaluation Metrics

-  Each model was evaluated using:
-  Accuracy
-  Precision
-  Recall
-  F1 Score
-  Confusion Matrix (visualized using Seaborn)

------------------------------------------------------------------------

## 🏆 Best Performing Model
For spam detection tasks, Naïve Bayes with TF-IDF typically performs the best and often achieves 97–98% accuracy.

Your notebook compares all models and identifies the top performer.

------------------------------------------------------------------------

## ▶️ How to Run

1️⃣ Install dependencies
```
pip install numpy pandas matplotlib seaborn nltk scikit-learn
```
2️⃣ Download NLTK requirements
```
nltk.download('stopwords')
nltk.download('wordnet')
```
3️⃣ Run the notebook
```
jupyter notebook "SPAM SMS DETECTION.ipynb"
```
------------------------------------------------------------------------
## 🔍 Example ML Pipeline
```
# Load dataset
df = pd.read_csv("spam.csv", usecols=['v1', 'v2'])
df.columns = ["label", "message"]

# Clean message text
df["clean_text"] = df["message"].apply(preprocess_text)

# Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

```
------------------------------------------------------------------------
## 🌟 Future Improvements
-  Deploy the model using Streamlit, Flask, or FastAPI
-  Integrate Word Embeddings (Word2Vec, GloVe)
-  Upgrade to deep learning models (LSTM/BERT)
-  Export as .pkl model and build a real-time SMS filter
  
-----------------------------------------------------------------------
## 👩‍💻 Author

**Monika A.D**
• AI & DS Student
(2025)

------------------------------------------------------------------------

Enjoy learning  🚀
