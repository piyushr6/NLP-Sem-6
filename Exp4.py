import nltk
from nltk.corpus import brown
from collections import defaultdict
import numpy as np

nltk.download('brown')
nltk.download('universal_tagset')

# Load Brown Corpus
tagged_sentences = brown.tagged_sents(tagset='universal')

# Compute Transition Counts
transition_counts = defaultdict(lambda: defaultdict(int))
tag_counts = defaultdict(int)

for sentence in tagged_sentences:
    prev_tag = "<START>"
    tag_counts[prev_tag] += 1

    for word, tag in sentence:
        transition_counts[prev_tag][tag] += 1
        tag_counts[tag] += 1
        prev_tag = tag

    transition_counts[prev_tag]["<END>"] += 1

# Compute Transition Probabilities
transition_prob = defaultdict(dict)

for prev_tag in transition_counts:
    total = sum(transition_counts[prev_tag].values())
    for tag in transition_counts[prev_tag]:
        transition_prob[prev_tag][tag] = transition_counts[prev_tag][tag] / total

# Compute Emission Counts
emission_counts = defaultdict(lambda: defaultdict(int))

for sentence in tagged_sentences:
    for word, tag in sentence:
        emission_counts[tag][word.lower()] += 1

# Compute Emission Probabilities
emission_prob = defaultdict(dict)

for tag in emission_counts:
    if tag == 0:
        continue
    total = sum(emission_counts[tag].values())
    for word in emission_counts[tag]:
        emission_prob[tag][word] = emission_counts[tag][word] / total

# Manual POS Tagging + Show Counts & Probabilities
def manual_hmm_tag(sentence, transition_prob, emission_prob, tag_counts,
                   emission_counts, transition_counts):

    words = sentence.split()
    tags = [tag for tag in tag_counts if tag not in ["<START>"]]

    predicted_tags = []
    prev_tag = "<START>"

    for word in words:
        word_lower = word.lower()
        best_tag = None
        max_prob = 0

        print(f"\nWORD: {word}")

        for tag in tags:
            e_count = emission_counts[tag].get(word_lower, 0)
            if e_count == 0:
                continue   

            # transition count
            t_count = transition_counts.get(prev_tag, {}).get(tag, 0)

            # probabilities
            emission = emission_prob[tag][word_lower]
            transition = transition_prob.get(prev_tag, {}).get(tag, 1e-6)

            prob = emission * transition

            print(f"Tag: {tag}")
            print(f"  Emission Count = {e_count}")
            print(f"  Transition Count ({prev_tag}->{tag}) = {t_count}")
            print(f"  Emission Prob = {emission}")
            print(f"  Transition Prob = {transition}")
            print(f"  Combined Prob = {prob}")

            if prob > max_prob:
                max_prob = prob
                best_tag = tag

        # if no tag found 
        if best_tag is None:
            best_tag = "NOUN"   

        predicted_tags.append((word, best_tag))
        prev_tag = best_tag

        print(f"Selected Tag: {best_tag}")

    return predicted_tags

# Test Sentence
test_sentence = "The quick brown fox jumps over the lazy dog"

manual_tags = manual_hmm_tag(
    test_sentence,
    transition_prob,
    emission_prob,
    tag_counts,
    emission_counts,
    transition_counts
)

print("\nManual HMM Tagging Result:")
for word, tag in manual_tags:
    print(word, "-", tag)

# NLTK HMM Tagger
from nltk.tag import hmm

trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(tagged_sentences)

nltk_tags = hmm_tagger.tag(test_sentence.split())

print("\nNLTK HMM Tagging:")
for word, tag in nltk_tags:
    print(word, "-", tag)