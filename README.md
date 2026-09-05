# Comment Category Prediction Challenge

A machine learning project to predict the category assigned to user-generated comments using textual, numerical, categorical, and metadata-based features.

---

## 📌 Project Overview

The **Comment Category Prediction Challenge** focuses on predicting the final category assigned to an online comment.

The dataset contains information about comments, engagement signals, emoticon indicators, internal system features, identity-related indicators, and timestamps. The objective is to explore these features, identify meaningful patterns, and build a classification model capable of accurately predicting the target `label`.

This project follows an end-to-end machine learning workflow:

**Data Exploration → Feature Engineering → Text Processing → Model Training → Hyperparameter Tuning → Model Stacking → Prediction → Submission**

---

## 🎯 Objective

The primary objective is to build a multiclass classification model that predicts the `label` associated with each comment.

The target variable contains **four distinct categories** representing different internal handling categories.

---

## 📂 Dataset

The project uses three main data files:

| File | Description |
|------|-------------|
| `train.csv` | Training dataset containing features and the target `label` |
| `test.csv` | Test dataset containing feature columns without the target |
| `sample_submission.csv` | Sample submission file showing the required submission format |

### Features

| Feature | Description |
|---------|-------------|
| `comment` | Raw text content of the comment |
| `created_date` | Date and time when the comment was posted |
| `post_id` | Identifier linking the comment to a discussion thread or parent post |
| `emoticon_1` | Indicator for the first internal emoticon group |
| `emoticon_2` | Indicator for the second internal emoticon group |
| `emoticon_3` | Indicator for the third internal emoticon group |
| `upvote` | Number of positive reactions |
| `downvote` | Number of negative reactions |
| `if_1` | Internal system feature 1 |
| `if_2` | Internal system feature 2 |
| `race` | Indicator for detected references to a specific group identity |
| `religion` | Indicator for detected references to belief-related topics |
| `gender` | Indicator for detected references to gender-related topics |
| `disability` | Indicator for detected references to ability-related topics |
| `label` | Target variable representing the final comment category |

---

## 🔍 Exploratory Data Analysis

The exploratory analysis investigates:

- Dataset structure and data types
- Missing values
- Statistical summaries
- Target-label distribution
- Correlations among numerical features
- Comment length and word count
- Relationship between identity-related features and labels
- Upvote and downvote patterns across categories
- Missing-value patterns in identity-related features

The analysis indicated that the numerical features generally have relatively weak correlations with each other and with the target. This motivated the inclusion of textual features as an important component of the predictive pipeline.

---

## 🛠️ Feature Engineering

Several additional features were created from the original dataset.

### Text-Based Features

- `text_len` — Number of characters in the comment
- `punc_count` — Count of `!` and `?`
- `caps_ratio` — Ratio of uppercase characters to comment length

### Engagement Features

- `vote_ratio` — Ratio of upvotes to downvotes
- `net_votes` — Difference between upvotes and downvotes

### Time-Based Features

The `created_date` field was converted into datetime format and used to derive:

- `hour` — Hour of the day when the comment was posted
- `is_weekend` — Indicates whether the comment was posted on a weekend

### Missing-Value Handling

Missing values in selected identity-related features were replaced with `"none"`.

The `comment` field was also converted to a string representation with missing values replaced by an empty string.

---

## 📝 Text Feature Extraction

The raw `comment` text is transformed using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

The text processing pipeline uses:

- Maximum of **15,000 features**
- Word n-grams ranging from **1 to 3**
- English stop-word removal
- Minimum document frequency of **3**

This allows the model to capture useful words as well as short phrases from the comments.

---

## ⚙️ Preprocessing Pipeline

A combined feature-processing pipeline was developed using `FeatureUnion`.

### Numerical Pipeline

The numerical features are processed using:

1. Feature selection
2. Median imputation
3. Standardization using `StandardScaler`

### Text Pipeline

The text features are processed using:

1. Comment extraction
2. TF-IDF vectorization
3. Unigram, bigram, and trigram features

The numerical and text representations are then combined before classification.

---

## 🤖 Machine Learning Models

Four baseline classification algorithms were evaluated:

- **Logistic Regression**
- **Random Forest**
- **XGBoost**
- **LightGBM**

The models were evaluated using the **weighted F1-score** on a stratified validation set.

### Train-Validation Split

The training dataset was divided into:

- **80% Training**
- **20% Validation**

A `random_state` of `42` was used for reproducibility, with stratification based on the target label.

---

## 🔧 Hyperparameter Tuning

Hyperparameter tuning was performed for:

- LightGBM
- XGBoost
- Random Forest

`RandomizedSearchCV` with **3-fold cross-validation** was used to explore different combinations of model parameters.

The tuning process optimized for **macro F1-score**.

---

## 🧩 Ensemble Learning

After tuning the individual models, a **Stacking Classifier** was created.

The ensemble combines:

- LightGBM
- XGBoost
- Random Forest
- Logistic Regression

A Logistic Regression model was used as the final estimator.

The purpose of stacking is to combine the strengths of different learning algorithms and potentially improve overall predictive performance.

---

## 🚀 Final Model Pipeline

The final workflow can be summarized as:

```text
                 ┌──────────────────┐
                 │    Raw Dataset   │
                 └────────┬─────────┘
                          │
                 ┌────────▼─────────┐
                 │ Feature Engineering│
                 └────────┬─────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
      ┌───────▼────────┐     ┌────────▼───────┐
      │ Numerical      │     │ Comment Text   │
      │ Features       │     │                │
      └───────┬────────┘     └────────┬───────┘
              │                       │
      Imputation +              TF-IDF Vectorization
      Standardization            (1–3 grams)
              │                       │
              └───────────┬───────────┘
                          │
                 ┌────────▼─────────┐
                 │ Combined Features│
                 └────────┬─────────┘
                          │
                 ┌────────▼─────────┐
                 │ Stacking Ensemble│
                 │                  │
                 │ LightGBM         │
                 │ XGBoost          │
                 │ Random Forest    │
                 │ Logistic Reg.    │
                 └────────┬─────────┘
                          │
                 ┌────────▼─────────┐
                 │ Final Prediction │
                 └──────────────────┘
```

## Submission

After training the final stacking model on the complete training dataset, predictions are generated for test.csv.

The submission file is created in the following format:

ID,label
1,category
2,category
3,category
...

The generated file is:   submission.csv
