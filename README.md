# Text Mining & NLP Projects
Welcome to my Text Mining repository. This repository contains various text mining projects.

## Introduction
This repository contains a comprehensive collection of Text Mining and Natural Language Processing (NLP) projects ranging from fundamental text processing to advanced Deep Learning applications. The goal of this repository is to demonstrate an end-to-end understanding of the NLP pipeline, including data cleansing, exploratory data analysis (EDA), feature extraction (TF-IDF, Bag-of-Words), and the implementation of machine learning models (Logistic Regression, Naive Bayes) as well as fine-tuning state-of-the-art transformers (BERT) for complex classification tasks.

These projects utilize unstructured data sources such as the **20 Newsgroups dataset** and **Goodreads Book Reviews** to derive actionable insights and automate decision-making processes.

---

## Project Summaries

### 1. `Text Processing.ipynb`
* **About:** This notebook serves as the foundation for handling textual data in Python.
* **Action:** Implemented core string manipulation techniques, including case conversion, string concatenation, and list slicing. It establishes the basics of transforming raw text strings into manageable data structures.
* **Goal Achieved:** Mastered the fundamental Python operations required to prepare and clean raw text data for downstream analysis.

### 2. `Regex, Spacy, Bag of Words.ipynb`
* **About:** A deep dive into pattern matching and advanced text normalization using industrial-strength NLP libraries.
* **Action:** Utilized **Regular Expressions (Regex)** to extract specific patterns (like emails or dates) and employed **spaCy** for linguistic processing such as tokenization. It also introduces the concept of **List Comprehensions** for efficient coding.
* **Goal Achieved:** Automated the extraction of specific information from unstructured text and prepared data for vectorization by normalizing linguistic structures.

### 3. `Bag of Words, TF_IDF, Text Similarity.ipynb`
* **About:** Focuses on converting text into numerical representations (Feature Extraction) and measuring document similarity.
* **Action:** Implemented **Bag-of-Words (BoW)** and **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization techniques. Calculated **Cosine Similarity** to find relationships between documents and built a **Naive Bayes classifier** to detect spam messages.
* **Goal Achieved:** Successfully transformed text into statistical vectors to perform quantitative analysis and built a functional text classification model.

### 4. `Exploratory Data Analysis & Topic Modeling.ipynb`
* **About:** A comprehensive analysis of the **20 Newsgroups dataset** to understand data distribution before modeling.
* **Action:** Performed rigorous EDA steps including checking for missing values, analyzing text length distributions via histograms, and visualizing high-frequency terms using **Word Clouds** and bar charts.
* **Goal Achieved:** Gained deep actionable insights into the dataset's structure and content, identifying key trends and data imbalances to inform model selection.

### 5. `Text_Classification.ipynb` & `AI for Humanists Fine Tuning Classification.ipynb`
* **About:** An advanced project focusing on Deep Learning and Transfer Learning for multi-class text classification using **Goodreads book reviews**.
* **Action:**
    * **Baseline:** Trained a Logistic Regression model using TF-IDF vectors as a benchmark.
    * **Deep Learning:** Fine-tuned a pre-trained **DistilBERT** transformer model using the **HuggingFace** library and **PyTorch**.
    * **Evaluation:** Compared model performance using precision, recall, and F1-scores, and analyzed misclassifications.
* **Goal Achieved:** Deployed a state-of-the-art NLP model that significantly outperforms traditional statistical methods in classifying book genres, demonstrating expertise in handling "Emerging Technologies" in AI.
