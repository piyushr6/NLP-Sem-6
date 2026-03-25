# Import Libraries
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

# Modified Abstracts
text_a = """Online learning systems generate large volumes of interaction data that can be used 
to understand student engagement. This data includes clickstream logs, quiz attempts, and time 
spent on resources. In this study, we propose a framework to analyze engagement patterns and 
identify at-risk learners. Experimental results show improved prediction accuracy compared to 
baseline models. The findings support early intervention strategies."""

text_b = """Text summarization aims to condense large documents into shorter versions while 
retaining important information. Recent advancements use transformer-based architectures for 
better contextual understanding. In this work, we design a hybrid summarization model combining 
extractive and abstractive methods. Evaluation using ROUGE metrics shows improved performance 
over traditional approaches. The model enhances readability and coherence."""


# Sentence Splitting Function
def break_into_sentences(text):
    parts = re.split(r'\.\s+', text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 6]

data_a = break_into_sentences(text_a)
data_b = break_into_sentences(text_b)

all_docs = data_a + data_b
targets = [0]*len(data_a) + [1]*len(data_b)

dataset = pd.DataFrame({
    "content": all_docs,
    "category": targets
})

dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

print("\n===== DATASET OVERVIEW =====")
print("Total Sentences:", len(dataset))
print(dataset.head(), "\n")

# Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

dataset["processed"] = dataset["content"].apply(clean_text)

# Vectorization
vec = CountVectorizer()
X = vec.fit_transform(dataset["processed"]).toarray()
y = dataset["category"].values

print("Vocabulary Count:", len(vec.get_feature_names_out()))

# Train-Test Split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=7
)

# Naive Bayes using Library
nb_model = MultinomialNB()
nb_model.fit(X_tr, y_tr)
pred_lib = nb_model.predict(X_te)

# Evaluation Function
def evaluate(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    TN, FP, FN, TP = cm.ravel()

    acc = (TP + TN) / (TP + TN + FP + FN)
    prec = TP / (TP + FP + 1e-9)
    rec = TP / (TP + FN + 1e-9)
    spec = TN / (TN + FP + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)

    print(f"\n===== {title} =====")
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {prec:.4f}")
    print(f"Recall       : {rec:.4f}")
    print(f"Specificity  : {spec:.4f}")
    print(f"F1 Score     : {f1:.4f}")

evaluate(y_te, pred_lib, "Library Naive Bayes Results")

# Test New Input
sample_text = ["Transformer models improve summarization quality significantly"]
sample_vec = vec.transform(sample_text)

print("\nLibrary Prediction:", nb_model.predict(sample_vec)[0])

# Scratch Implementation
class CustomNB:
    def fit(self, X, y):
        self.labels = np.unique(y)
        self.priors = {}
        self.likelihoods = {}

        for label in self.labels:
            X_c = X[y == label]
            self.priors[label] = len(X_c) / len(X)

            word_freq = np.sum(X_c, axis=0) + 1
            total = np.sum(word_freq)
            self.likelihoods[label] = word_freq / total

    def predict(self, X):
        results = []
        for row in X:
            scores = {}
            for label in self.labels:
                log_prob = np.log(self.priors[label])
                log_prob += np.sum(row * np.log(self.likelihoods[label]))
                scores[label] = log_prob

            results.append(max(scores, key=scores.get))

        return np.array(results)

# Train Scratch Model
custom_nb = CustomNB()
custom_nb.fit(X_tr, y_tr)

pred_scratch = custom_nb.predict(X_te)

# Evaluate Scratch Model
evaluate(y_te, pred_scratch, "Scratch Naive Bayes Results")

# Prediction (Scratch)
print("\nScratch Prediction:", custom_nb.predict(sample_vec.toarray())[0])