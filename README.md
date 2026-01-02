# Quora Duplicate Question Detection

This repository contains multiple machine learning approaches to identify whether two questions from the **Quora Question Pairs dataset** are duplicates or not. The project explores how different **feature engineering (FE)** and **text representation techniques** impact model performance.

---

## 📌 Problem Statement

Given a pair of questions (Q1, Q2), the goal is to predict whether they are semantically equivalent (duplicate) or not.

This is a classic **binary text classification** problem widely used to benchmark NLP pipelines.

---

## 📊 Approaches & Results

The following experiments were conducted with different feature representations and models:

| # | Features / Representation                       | Model         | Accuracy |
| - | ----------------------------------------------- | ------------- | -------- |
| 1 | Bag of Words (no preprocessing)                 | Random Forest | **75%**  |
| 2 | Preprocessing + TF-IDF                          | XGBoost       | **72%**  |
| 3 | Custom Features + Bag of Words                  | Random Forest | **77%**  |
| 4 | Custom Features + Sentence Transformer + TF-IDF | XGBoost       | **82%**  |

📌 **Best Performance:** Combination of **custom features + semantic embeddings + TF-IDF**

---

## 🧠 Feature Engineering

### 🔹 Text Preprocessing

* Lowercasing
* Punctuation removal
* Stopword removal
* URL removal
* Stemming
* Decontracting words
* Slang normalization

### 🔹 Custom Features

The following handcrafted features were used:

* `q1_len` – Character length of question 1
* `q2_len` – Character length of question 2
* `q1_word_count` – Number of words in question 1
* `q2_word_count` – Number of words in question 2
* `common_word_count` – Number of common words between Q1 and Q2
* `total_word_count` – Total unique words across both questions
* `word_share` – Ratio of common words to total words

These features help capture **lexical similarity** beyond raw text vectors.

---

## 🔤 Text Representation Techniques

* **Bag of Words (BoW)**
* **TF-IDF Vectorization**
* **Sentence Transformers** (`all-MiniLM-L6-v2`)

Sentence embeddings help capture **semantic similarity**, which significantly improved performance.

---

## 🤖 Models Used

* Random Forest Classifier
* XGBoost Classifier

---



## ⚙️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Sentence Transformers
* NLTK / Regex

---

## 🚀 Key Learnings

* Preprocessing alone does not guarantee better performance
* Handcrafted features significantly boost classical ML models
* Combining **semantic embeddings with statistical features** yields the best results
* Tree-based models benefit strongly from engineered numerical features

---

## 📈 Future Improvements

* Fine-tune transformer models (BERT, RoBERTa)
* Use cosine similarity directly on embeddings
* Handle class imbalance with advanced sampling

---

## 🙌 Acknowledgements

* Quora Question Pairs Dataset
* HuggingFace Sentence Transformers

---

