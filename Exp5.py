import nltk
import time
import itertools
from nltk.corpus import treebank
from nltk.tag import hmm as nltk_hmm
from collections import defaultdict

nltk.download("treebank", quiet=True)

print("\nLoading Penn Treebank...\n")

tagged_sentences = treebank.tagged_sents(tagset="universal")

train_data = tagged_sentences[200:]
test_data  = tagged_sentences[:200]


# -----------------------------
# Build HMM probabilities
# -----------------------------

transition_counts = defaultdict(lambda: defaultdict(int))
emission_counts   = defaultdict(lambda: defaultdict(int))

for sentence in train_data:
    prev = "<START>"

    for word, tag in sentence:
        transition_counts[prev][tag] += 1
        emission_counts[tag][word.lower()] += 1
        prev = tag

    transition_counts[prev]["<END>"] += 1


transition_prob = defaultdict(dict)
emission_prob   = defaultdict(dict)

for prev, nexts in transition_counts.items():
    total = sum(nexts.values())
    for tag, cnt in nexts.items():
        transition_prob[prev][tag] = cnt / total


for tag, words in emission_counts.items():
    total = sum(words.values())
    for word, cnt in words.items():
        emission_prob[tag][word] = cnt / total


tags = list(emission_prob.keys())

print("Tags learned    :", len(tags))
print("Training sents  :", len(train_data))
print("Test sents      :", len(test_data))
print()


# --------------------------------
# Settings
# --------------------------------

MAX_LEN = 5          # sentence length used for comparison
BF_TAG_LIMIT = 8     # limit tags for brute force


# --------------------------------
# Sequence scoring
# --------------------------------

def sequence_score(words, tag_sequence):

    score = transition_prob["<START>"].get(tag_sequence[0], 1e-10)
    score *= emission_prob[tag_sequence[0]].get(words[0], 1e-10)

    for i in range(1, len(words)):

        score *= transition_prob.get(tag_sequence[i-1], {}).get(tag_sequence[i], 1e-10)
        score *= emission_prob[tag_sequence[i]].get(words[i], 1e-10)

    score *= transition_prob.get(tag_sequence[-1], {}).get("<END>", 1e-10)

    return score


# --------------------------------
# Brute Force Decoder
# --------------------------------

def brute_force_hmm(sentence):

    words = [w.lower() for w in sentence]

    n = len(words)

    best_score = -1
    best_seq = None

    candidate_tags = tags[:BF_TAG_LIMIT]

    for tag_seq in itertools.product(candidate_tags, repeat=n):

        score = sequence_score(words, tag_seq)

        if score > best_score:
            best_score = score
            best_seq = tag_seq

    return list(best_seq)


# --------------------------------
# Viterbi Decoder
# --------------------------------

def viterbi_manual(sentence):

    V = [{}]
    B = [{}]

    first_word = sentence[0].lower()

    for tag in tags:

        V[0][tag] = (
            transition_prob["<START>"].get(tag, 1e-10) *
            emission_prob[tag].get(first_word, 1e-10)
        )

        B[0][tag] = "<START>"


    for t in range(1, len(sentence)):

        V.append({})
        B.append({})

        word = sentence[t].lower()

        for curr_tag in tags:

            emission = emission_prob[curr_tag].get(word, 1e-10)

            max_prob = -1
            best_prev = None

            for prev_tag in tags:

                prob = (
                    V[t-1].get(prev_tag, 0) *
                    transition_prob.get(prev_tag, {}).get(curr_tag, 0) *
                    emission
                )

                if prob > max_prob:
                    max_prob = prob
                    best_prev = prev_tag

            V[t][curr_tag] = max_prob
            B[t][curr_tag] = best_prev


    last_index = len(sentence) - 1

    best_last_tag = max(V[last_index], key=V[last_index].get)

    best_path = [best_last_tag]

    for t in range(last_index, 0, -1):

        best_last_tag = B[t][best_last_tag]
        best_path.insert(0, best_last_tag)

    return best_path


# --------------------------------
# Evaluation Function
# --------------------------------

def evaluate(decoder_fn, label):

    total = 0
    correct = 0

    start = time.time()

    for sentence in test_data:

        words = [w for w, _ in sentence][:MAX_LEN]
        gold  = [t for _, t in sentence][:MAX_LEN]

        pred = decoder_fn(words)

        for p, g in zip(pred, gold):

            total += 1

            if p == g:
                correct += 1

    elapsed = time.time() - start

    acc = correct / total

    print(f"{label:<40} Accuracy: {acc:.4f} | Time: {elapsed:.2f}s")

    return acc, elapsed


# --------------------------------
# PART 1 : Brute Force vs Viterbi
# --------------------------------

print("="*60)
print("Manual HMM : Brute Force vs Viterbi")
print(f"Sentence length used for comparison : {MAX_LEN}")
print("="*60)
print()


bf_acc, bf_time = evaluate(
    brute_force_hmm,
    "Manual HMM + Brute Force"
)

vt_acc, vt_time = evaluate(
    viterbi_manual,
    "Manual HMM + Viterbi"
)


print()

print("Accuracy Difference (Viterbi - Brute Force) :", round((vt_acc-bf_acc)*100,4), "%")
print("Speed-up of Viterbi :", round(bf_time/vt_time,1), "x faster")


# --------------------------------
# PART 2 : NLTK HMM
# --------------------------------

print("\n")
print("="*60)
print("NLTK Library HMM + Viterbi")
print("="*60)
print()

trainer = nltk_hmm.HiddenMarkovModelTrainer()

nltk_model = trainer.train(train_data)

nltk_total = 0
nltk_correct = 0

start = time.time()

for sentence in test_data:

    words = [w for w,_ in sentence]
    gold  = [t for _,t in sentence]

    pred = nltk_model.tag(words)

    for (_,p),g in zip(pred,gold):

        nltk_total += 1

        if p == g:
            nltk_correct += 1


nltk_time = time.time() - start
nltk_acc = nltk_correct/nltk_total


print(f"{'NLTK HMM + Viterbi':<40} Accuracy: {nltk_acc:.4f} | Time: {nltk_time:.2f}s")


# --------------------------------
# Example sentence
# --------------------------------

print("\n")
print("="*60)
print("Example Tagging")
print("="*60)

test_sentence = ["The","stock","market","fell","sharply"]

bf_tags = brute_force_hmm(test_sentence[:MAX_LEN])
vt_tags = viterbi_manual(test_sentence[:MAX_LEN])
nltk_tags = [t for _,t in nltk_model.tag(test_sentence)]


print("\nSentence :",test_sentence,"\n")

print(f"{'Word':<12}{'BruteForce':<15}{'Viterbi':<15}{'NLTK'}")
print("-"*50)

for w,b,v,n in zip(test_sentence,bf_tags,vt_tags,nltk_tags):

    print(f"{w:<12}{b:<15}{v:<15}{n}")


# --------------------------------
# Final Summary
# --------------------------------

print("\n")
print("="*60)
print("FINAL SUMMARY")
print("="*60)

print(f"{'Method':<42}{'Accuracy':>10}{'Time':>10}")
print("-"*60)

print(f"{'Manual HMM + Brute Force':<42}{bf_acc:>10.4f}{bf_time:>10.2f}s")
print(f"{'Manual HMM + Viterbi':<42}{vt_acc:>10.4f}{vt_time:>10.2f}s")
print(f"{'NLTK HMM + Viterbi':<42}{nltk_acc:>10.4f}{nltk_time:>10.2f}s")

print("="*60)


print("\nTheoretical Complexity")
print("Brute Force : O(N^T)")
print("Viterbi     : O(T * N^2)")       