# =========================
# Experiment 1 – NLP Basics
# =========================

import matplotlib.pyplot as plt
import nltk
import spacy

# Download required NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# -------------------------
# Input Paragraph
# -------------------------
paragraph = (
    "Students are studying every day because studying regularly helps students learn better. "
    "Some students learn quickly, while other students are learning slowly and need more learning time. "
    "Teachers are teaching lessons, teachers teach concepts clearly, and good teachers have taught students well. "
    "Learning is not only about reading books but also about reading, writing, and thinking deeply. "
    "When students are running experiments, they run tests again and again because repeated running of tests "
    "makes results better and better. The best students are those who keep working, have worked hard, "
    "and continue working even when the work is difficult."
)

# =====================================================
# Frequency Distribution – WITHOUT using library
# =====================================================
freq = {}

for word in paragraph.split():
    word = word.lower().strip(".,")
    if word in freq:
        freq[word] += 1
    else:
        freq[word] = 1

print("\nFrequency Distribution (Without Library):")
for word, count in freq.items():
    print(f"{word}: {count}")

plt.figure(figsize=(10, 5))
plt.bar(freq.keys(), freq.values())
plt.xticks(rotation=70)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Word Frequency Distribution (Without Library)")
plt.tight_layout()
plt.show()

# =====================================================
# Frequency Distribution – WITH NLTK
# =====================================================
from nltk.probability import FreqDist

words = nltk.word_tokenize(paragraph.lower())
fdist = FreqDist(words)

print("\nFrequency Distribution (With NLTK):")
for word, count in fdist.items():
    print(f"{word}: {count}")

plt.figure(figsize=(10, 5))
plt.bar(fdist.keys(), fdist.values())
plt.xticks(rotation=70)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Word Frequency Distribution (With NLTK)")
plt.tight_layout()
plt.show()

# =====================================================
# Tokenization
# =====================================================
from nltk.tokenize import word_tokenize
tokens = word_tokenize(paragraph)

print("\nTokens:")
print(tokens)

# =====================================================
# Stop-word Removal
# =====================================================
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

filtered_words = [
    word.lower()
    for word in tokens
    if word.isalnum() and word.lower() not in stop_words
]

print("\nAfter Stop-word Removal:")
print(filtered_words)

# =====================================================
# Add–Delete Table with Morphological Features
# =====================================================
root_word = input("\nEnter root word: ").lower()

derived_words = [
    root_word + "s",
    root_word + "ed",
    root_word + "ing",
    root_word + "er",
    root_word + "ly"
]

print("\nAdd–Delete Table with Morphological Features")
print("{:<15} {:<10} {:<10} {:<15} {:<20} {:<10}".format(
    "Word", "Deleted", "Added", "Singular", "Tense", "Gender"
))
print("-" * 85)

for word in derived_words:
    doc = nlp(word)
    lemma = doc[0].lemma_

    # Add–Delete
    deleted = "-"
    added = word.replace(root_word, "")

    # Singular form
    singular_form = lemma

    # Tense detection
    if word.endswith("ed"):
        tense = "Past Tense"
    elif word.endswith("ing"):
        tense = "Present Participle"
    elif word.endswith("s"):
        tense = "Present (3rd Person)"
    else:
        tense = "Base Form"

    # Gender (not applicable in English morphology)
    gender = "N/A"

    print("{:<15} {:<10} {:<10} {:<15} {:<20} {:<10}".format(
        word, deleted, added, singular_form, tense, gender
    ))
