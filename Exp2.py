import nltk
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer
from nltk.corpus import stopwords
import string

para = "The movie delivers stunning visuals, complex ideas, intense action, and emotional moments. Characters plan missions, enter dreams, fight enemies, and chase memories. The brilliant direction builds suspense, twists reality, and explores ambition, guilt, love, and hope, creating a powerful cinematic experience that grips audiences and sparks imagination."

stop_words = set(stopwords.words('english'))

#remove punctuation + lowercase
para = para.translate(str.maketrans('', '', string.punctuation)).lower()

#split into words
words = para.split()

#remove stopwords
filtered_words = [w for w in words if w not in stop_words]
print(filtered_words)

#initialize stemmers
porter = PorterStemmer()
snowball = SnowballStemmer("english")
lancaster = LancasterStemmer()

print(f"{'Word':<15}{'Porter':<15}{'Snowball':<15}{'Lancaster':<15}")
print("-" * 60)

for w in words:
    p = porter.stem(w)
    s = snowball.stem(w)
    l = lancaster.stem(w)

    print(f"{w:<15}{p:<15}{s:<15}{l:<15}")


print("************************************")
print("************************************")
print("************************************")


# Stemming vs Lemmatization

from nltk.stem import WordNetLemmatizer
import string

nltk.download('wordnet')
nltk.download('omw-1.4')
porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()

#clean punctuation + lowercase
para = para.translate(str.maketrans('', '', string.punctuation)).lower()
words = para.split()

print(f"{'Word':<15}{'Stem':<15}{'Lemma':<15}")
print("-" * 45)

for w in words:
    stem = porter.stem(w)
    lemma = lemmatizer.lemmatize(w)

    print(f"{w:<15}{stem:<15}{lemma:<15}")


print("************************************")
print("************************************")
print("************************************")




# Implementing Porter Stemmer from Scratch (WITHOUT LIBRARY)
import nltk
from nltk.corpus import treebank
import re

nltk.download("treebank")


def is_vowel(ch):
    return ch in "aeiou"

def contains_vowel(word):
    for c in word:
        if is_vowel(c):
            return True
    return False

def ends_double(word):
    return len(word) > 1 and word[-1] == word[-2]

def cvc(word):
    if len(word) < 3:
        return False
    c1, v, c2 = word[-3], word[-2], word[-1]
    return (not is_vowel(c1)) and is_vowel(v) and (not is_vowel(c2)) and c2 not in "wxy"

def measure(word):
    pattern = re.compile(r"[aeiou]+[^aeiou]+")
    return len(pattern.findall(word))



def porter_stem(word):

    word = word.lower()

    # Step 1a - short suffixes
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]

    # Step 1b 
    if word.endswith("eed"):
        stem = word[:-3]
        if measure(stem) > 0:
            word = stem + "ee"

    elif word.endswith("ed"):
        stem = word[:-2]
        if contains_vowel(stem):
            word = stem

    elif word.endswith("ing"):
        stem = word[:-3]
        if contains_vowel(stem):
            word = stem

    if word.endswith(("at","bl","iz")):
        word += "e"
    elif ends_double(word) and word[-1] not in "lsz":
        word = word[:-1]
    elif measure(word) == 1 and cvc(word):
        word += "e"

    # Step 1c
    if word.endswith("y"):
        stem = word[:-1]
        if contains_vowel(stem):
            word = stem + "i"

    # Step 2 - long suffixes
    step2 = {
        "ational":"ate","tional":"tion","enci":"ence","anci":"ance",
        "izer":"ize","abli":"able","alli":"al","entli":"ent","eli":"e",
        "ousli":"ous","ization":"ize","ation":"ate","ator":"ate",
        "alism":"al","iveness":"ive","fulness":"ful","ousness":"ous",
        "aliti":"al","iviti":"ive","biliti":"ble"
    }
   # replace
    for k in step2:
        if word.endswith(k):
            stem = word[:-len(k)]
            if measure(stem) > 0:
                word = stem + step2[k]
            break

    # Step 3 - cleanup
    step3 = {
        "icate":"ic","ative":"","alize":"al",
        "iciti":"ic","ical":"ic","ful":"","ness":""
    }

    for k in step3:
        if word.endswith(k):
            stem = word[:-len(k)]
            if measure(stem) > 0:
                word = stem + step3[k]
            break

    # Step 4
    step4 = ["al","ance","ence","er","ic","able","ible","ant","ement",
             "ment","ent","ion","ou","ism","ate","iti","ous","ive","ize"]

    for s in step4:
        if word.endswith(s):
            stem = word[:-len(s)]
            if measure(stem) > 1: #at least 2 VC pairs
                if s != "ion" or stem.endswith(("s","t")): #eg revision -> revis, but adoption -> adopt(ion)
                    word = stem
            break

    # Step 5a
    if word.endswith("e"):
        stem = word[:-1]  #eg probate -> probat
        if measure(stem) > 1 or (measure(stem)==1 and not cvc(stem)):
            word = stem

    # Step 5b
    if measure(word) > 1 and ends_double(word) and word.endswith("l"): 
        word = word[:-1] #eg parallel -> paralel

    return word

#apply on penn treebank words
words = [w.lower() for w in treebank.words() if w.isalpha()]
words = list(dict.fromkeys(words))[:30]

print("Word\t\tStem")
print("----------------------")

for w in words:
    print(f"{w:<12}\t{porter_stem(w)}")