# 🎵 Music Popularity Prediction with Multi-Modal Analysis

This repository contains the code, data, and report for our class project: **Multi-Modal Analysis of Audio, Artist, and Lyrical Features to Predict Music Popularity**.

We aimed to understand what makes a song popular by analyzing diverse data inputs such as audio features, artist metadata, and lyrical content using both traditional regression and neural network models. Our best model was a neural network that integrated these features, outperforming traditional models.

---

## Project Overview

- **Goal:** Predict a song’s popularity score using audio features, artist metadata, and lyrical sentiment/topic analysis.
- **Dataset:** Spotify Huge Database – Daily Charts Over 3 Years (from Kaggle).
- **Approach:**
  - **Data Cleaning & Enrichment:** Collect, clean, and standardize audio features and artist metadata. Scrape lyrics via Genius API and translate non-English lyrics.
  - **Feature Engineering:** LDA topic modeling and sentiment analysis for lyrics.
  - **Modeling:** Baseline regression models, regularized models, Kernel Ridge Regression, and custom neural networks (PyTorch & scikit-learn).
  - **Results:** Neural networks outperformed traditional regression, showing the importance of non-linear modeling for song popularity.

---

## 📂 Repository Structure

```
ML_Popularity_Prediction/
│
├── notebooks/
│   ├── Model_Comparisons.ipynb            # Main ML analysis and model comparisons
│   ├── Natural_Language_Analysis.ipynb    # Lyrics preprocessing and topic modeling
│   └── Sentiment_Analysis_Pytorch.ipynb   # Sentiment analysis with logistic regression
│
├── data/
│   └── FINAL_CLEAN_SPOTIFY_DATA.csv       # Cleaned dataset for model training/testing
│
├── ML_Final_Project.pdf                              # Research paper summarizing our methods and findings
│
├── README.md                              # Project overview and instructions
│
└── requirements.txt                       # Python dependencies
```

---

## 🔍 How to Use

1. **Clone this repository:**
   ```bash
   git clone https://github.com/yourusername/ML_Popularity_Prediction.git
   cd ML_Popularity_Prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the raw Spotify dataset (if desired):**
   - Due to size and licensing, we do **not** include the raw dataset.
   - Download it manually from [Kaggle: Spotify Huge Database](https://www.kaggle.com/datasets/pepepython/spotify-huge-database-daily-charts-over-3-years).
   - Follow instructions in the `notebooks/Natural_Language_Analysis.ipynb` and `Sentiment_Analysis_Pytorch.ipynb` to preprocess the raw data into the final cleaned dataset.

4. **Run the notebooks:**
   - **Step 1:** `notebooks/Natural_Language_Analysis.ipynb` — performs lyrical feature engineering.
   - **Step 2:** `notebooks/Sentiment_Analysis_Pytorch.ipynb` — computes sentiment scores.
   - **Step 3:** `notebooks/Model_Comparisons.ipynb` — trains and compares models using the cleaned dataset.

---

## 📊 Results Summary

| Model                    | Test MSE  | Test R²   |
|--------------------------|----------|-----------|
| Linear Regression        | 0.2214   | 0.0938    |
| Ridge Regression         | 0.2214   | 0.0938    |
| Lasso Regression         | 0.2429   | 0.0057    |
| ElasticNet Regression    | 0.2316   | 0.0519    |
| Kernel Ridge Regression  | 0.1905   | 0.2201    |
| **Neural Network (PyTorch)**      | 0.1733   | 0.2905    |
| **Neural Network (scikit-learn)** | **0.1702** | **0.3034** |

Our neural networks significantly outperformed traditional models, highlighting the value of deep learning in capturing complex relationships between song features and popularity.

---

## 📑 Research Paper

Please refer to `docs/ML_Final_Project.pdf` for detailed methodology, implementation details, and result analysis.

---

## 📝 License

This project uses datasets under the Kaggle license—please ensure compliance when using this data. The code is open for educational and non-commercial use.

---

## 🤝 Contributors

- [Jennifer Chen](https://github.com/ChenJieNi2004)
- [Sophia Dai](https://github.com/blubmeowfishycat)
- [Leslie Kim](https://github.com/Leslie714)
- [Rose Zhao](https://github.com/rose-zz)

---

