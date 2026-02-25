from collections import defaultdict

# --------------------------------------------------
# IDEAL 4-SENTENCE CORPUS WITH <s> </s>
# --------------------------------------------------

paragraph = """
<s> machine learning systems learn from data </s>
<s> machine learning models learn from data </s>
<s> artificial intelligence systems use machine learning </s>
<s> machine learning systems make predictions from data </s>
"""

tokens = paragraph.lower().split()

# --------------------------------------------------
# COUNTS
# --------------------------------------------------

unigram = defaultdict(int)
bigram = defaultdict(int)
trigram = defaultdict(int)
vocab = set()

for i in range(len(tokens)):
    unigram[tokens[i]] += 1
    vocab.add(tokens[i])

    if i < len(tokens)-1:
        bigram[(tokens[i], tokens[i+1])] += 1

    if i < len(tokens)-2:
        trigram[(tokens[i], tokens[i+1], tokens[i+2])] += 1

vocab = sorted(vocab)
V = len(vocab)

print("Vocabulary:", vocab)
print("Vocabulary size:", V)

# --------------------------------------------------
# BIGRAM SPARSE TABLE
# --------------------------------------------------

print("\nBIGRAM SPARSE TABLE\n")
for k,v in bigram.items():
    print(k,":",v)

# --------------------------------------------------
# BIGRAM COUNT MATRIX
# --------------------------------------------------

print("\nBIGRAM COUNT MATRIX\n")

print("     ", end="")
for w in vocab:
    print(f"{w[:6]:>7}", end="")
print()

for w1 in vocab:
    print(f"{w1[:6]:>6}", end=" ")
    for w2 in vocab:
        print(f"{bigram[(w1,w2)]:>7}", end="")
    print()

# --------------------------------------------------
# ADD-ONE SMOOTHING
# --------------------------------------------------

def bigram_prob(w1,w2):
    return (bigram[(w1,w2)] + 1) / (unigram[w1] + V)

def trigram_prob(w1,w2,w3):
    return (trigram[(w1,w2,w3)] + 1) / (bigram[(w1,w2)] + V)

# --------------------------------------------------
# BIGRAM PROBABILITY TABLE
# --------------------------------------------------

print("\nBIGRAM PROBABILITY TABLE (ADD-ONE)\n")

for w1 in vocab:
    for w2 in vocab:
        print(f"P({w2}|{w1}) = {bigram_prob(w1,w2):.4f}")
    print()

# --------------------------------------------------
# TRIGRAM SPARSE TABLE
# --------------------------------------------------

print("\nTRIGRAM SPARSE TABLE\n")

for k,v in trigram.items():
    print(k,":",v)

# --------------------------------------------------
# TRIGRAM PROBABILITY TABLE (SPARSE)
# --------------------------------------------------

print("\nTRIGRAM PROBABILITY TABLE\n")

for k in trigram:
    w1,w2,w3 = k
    print(f"P({w3}|{w1},{w2}) = {trigram_prob(w1,w2,w3):.4f}")

# --------------------------------------------------
# SENTENCE PROBABILITY + VALIDITY (UPDATED)
# --------------------------------------------------

def bigram_sentence(sent):
    w = sent.lower().split()
    p = 1
    for i in range(len(w)-1):
        p *= bigram_prob(w[i],w[i+1])
    return p

def trigram_sentence(sent):
    w = sent.lower().split()
    p = 1
    for i in range(len(w)-2):
        p *= trigram_prob(w[i],w[i+1],w[i+2])
    return p

def valid_bigram(sent):
    w = sent.lower().split()
    for i in range(len(w)-1):
        if bigram[(w[i],w[i+1])] == 0:
            return False
    return True

def valid_trigram(sent):
    w = sent.lower().split()
    for i in range(len(w)-2):
        if trigram[(w[i],w[i+1],w[i+2])] == 0:
            return False
    return True

# --------------------------------------------------
# ORDER BASED VALID / INVALID
# --------------------------------------------------

valid_sentence = "<s> machine learning systems learn from data </s>"
invalid_sentence = "<s> systems learning machine learn from data </s>"

print("\n-------- VALID ORDER SENTENCE --------")
print(valid_sentence)

print("Bigram Probability :", bigram_sentence(valid_sentence))
print("Trigram Probability:", trigram_sentence(valid_sentence))

print("Bigram Valid:", valid_bigram(valid_sentence))
print("Trigram Valid:", valid_trigram(valid_sentence))

print("\n-------- INVALID ORDER SENTENCE --------")
print(invalid_sentence)

print("Bigram Probability :", bigram_sentence(invalid_sentence))
print("Trigram Probability:", trigram_sentence(invalid_sentence))

print("Bigram Valid:", valid_bigram(invalid_sentence))
print("Trigram Valid:", valid_trigram(invalid_sentence))