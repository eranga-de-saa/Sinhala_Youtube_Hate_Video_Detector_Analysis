# Hate Content Detection in Sinhala YouTube Videos

This repository contains the implementation of a **Self-Reflective and Introspective Feature Model** for detecting hate content in Sinhala-language YouTube videos using Natural Language Processing (NLP) and supervised machine learning techniques.

## 📖 Overview

The system classifies YouTube videos as hateful or non-hateful based on:

* User comments
* Thumbnail text
* Video metadata (title, description, tags)
* Like/dislike ratios
* Positive/negative comment ratios

The model achieves **\~89% accuracy** using a Logistic Regression classifier with TF-IDF vectorization and handcrafted linguistic features.

## 🗂️ Dataset

* **Video Dataset:** 1,000 Sinhala YouTube videos (450 hateful, 550 non-hateful)
* **Comment Dataset:** 2,000 user comments (1,050 positive, 950 negative)

Data was manually annotated using majority voting by three annotators based on YouTube’s hate speech policy.

## 🛠️ Methodology

### 1. Preprocessing

* Removal of hashtags, punctuation, emojis
* Tokenization using NLTK
* Part-of-Speech tagging (using UCSC POS dataset)
* Stemming

### 2. Negation Handling

Handled Sinhala negators like **නැ, නෑ, නැහැ, නැත, බැ, බෑ, බැහැ, බැය, එපා** by prefixing affected words with `not_`.

### 3. Feature Extraction

* **Text Vectorization:** TF-IDF, Count Vectorization, Word2Vec
* **Lexicon-based Features:** Positive/negative word percentages
* **Metadata Features:** Like/dislike ratio, positive/negative comment ratio

### 4. Classification Models

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* Artificial Neural Network (ANN)
* Multinomial Naive Bayes (MNB)

## 📊 Results

### Comment Sentiment Analysis

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 77%      |
| SVM                 | 77%      |
| Random Forest       | 75%      |
| ANN                 | 74%      |
| Lexicon-based       | 52%      |

### Hate Detection

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 89%      |
| Multinomial NB      | 83%      |
| Random Forest       | 83%      |
| ANN                 | 81%      |

## 🚀 Installation & Usage

### Prerequisites

* Python 3.7+
* Libraries: `nltk`, `scikit-learn`, `pandas`, `numpy`, `tensorflow` (optional for ANN)

### Steps

Clone the repository:

```bash
git clone https://github.com/your-username/sinhala-hate-detection.git
cd sinhala-hate-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the preprocessing and training scripts:

```bash
python preprocess.py
python train_comment_classifier.py
python train_hate_detector.py
```

Evaluate models:

```bash
python evaluate.py
```

## 📌 Future Work

* Sarcasm detection in comments
* Handling more Sinhala negation types
* Expanding the dataset for better generalization
* Cross-platform validation (e.g., Facebook, Twitter)

## 🙏 Acknowledgments

This research was supported by the **Accelerating Higher Education Expansion and Development (AHEAD) Operation** of the Ministry of Higher Education, Sri Lanka, funded by the World Bank.

## 📜 Citation

If you use this work, please cite:

```bibtex
@inproceedings{desaa2020self,
  title={Self-Reflective and Introspective Feature Model for Hate Content Detection in Sinhala YouTube Videos},
  author={De Saa, Eranga and Ranathunga, Lochandaka},
  booktitle={2020 IEEE Conference},
  pages={1--6},
  year={2020},
  organization={IEEE}
}
```

## 📄 License

This project is licensed under the **MIT License**. See LICENSE for details.

## 📧 Contact

For questions or collaborations, feel free to reach out:

* **Eranga De Saa:** [erangdesaa@gmail.com](mailto:erangdesaa@gmail.com)
* **Lochandaka Ranathunga:** [lochandaka@uom.lk](mailto:lochandaka@uom.lk)

---

*Disclaimer: This project is intended for research purposes. Always comply with YouTube's Terms of Service and API usage policies when collecting data.*
