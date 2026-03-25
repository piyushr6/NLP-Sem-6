# ───────────────────────────────────────────────────────────────
#  INSTALL COMMANDS (run once in terminal before executing):
#
#  pip install spacy
#  python -m spacy download en_core_web_sm
#  pip install sklearn-crfsuite

# PART 1 : Using a pre-trained library (spaCy) for NER
# --------------------------------------------------------------
import spacy 

def ner_with_library(text: str):
   
    # Load the small English model (downloaded once via the command above).
    # It bundles tokeniser + tagger + parser + NER in one pipeline.
    nlp = spacy.load("en_core_web_sm")

    # Process the raw text — this runs every pipeline component
    doc = nlp(text)

    print("\n" + "=" * 60)
    print("  PART 1 — NER WITH spaCy LIBRARY")
    print("=" * 60)
    print(f"\nInput Text:\n  \"{text}\"\n")
    print(f"{'Entity Text':<30} {'Label':<20} {'Description'}")
    print("-" * 70)

    # doc.ents is a tuple of Span objects — each Span is one entity
    for entity in doc.ents:
        description = spacy.explain(entity.label_) or "—"
        print(f"{entity.text:<30} {entity.label_:<20} {description}")
  
    return [(ent.text, ent.label_) for ent in doc.ents]


#  PART 2 — NER WITHOUT ANY LIBRARY
# --------------------------------------------------

import re                      
import string

PERSON_LIST = {
    "tim cook", "elon musk", "sundar pichai", "jeff bezos",
    "bill gates", "mark zuckerberg", "satya nadella",
    "steve jobs", "warren buffett", "barack obama",
    "narendra modi", "ratan tata", "sam altman"
}

ORGANIZATION_LIST = {
    "apple", "google", "microsoft", "amazon", "meta",
    "tesla", "openai", "anthropic", "infosys", "tata",
    "reliance", "ibm", "intel", "samsung", "twitter",
    "facebook", "netflix", "spotify", "uber", "airbnb"
}

LOCATION_LIST = {
    "india", "usa", "california", "new york", "london",
    "paris", "tokyo", "beijing", "mumbai", "delhi",
    "cupertino", "seattle", "bangalore", "hyderabad",
    "san francisco", "los angeles", "chicago", "boston",
    "singapore", "dubai", "berlin", "sydney", "toronto"
}

DATE_PATTERNS = [
    # e.g. "January 5, 2023" or "5 January 2023"
    r"\b(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
    r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|"
    r"August|September|October|November|December)\s+\d{4}\b",
    # e.g. "2023-01-05" or "01/05/2023"
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b\d{2}/\d{2}/\d{4}\b",
    # Relative: "last Monday", "next year"
    r"\b(?:last|next|this)\s+(?:Monday|Tuesday|Wednesday|Thursday|"
    r"Friday|Saturday|Sunday|week|month|year)\b",
]

MONEY_PATTERNS = [
    # e.g. "$500", "₹1,000", "€200 million"
    r"[$₹€£¥]\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|trillion))?",
    r"\b\d[\d,]*(?:\.\d+)?\s?(?:dollars?|rupees?|euros?|pounds?|USD|INR|EUR)\b",
]

EMAIL_PATTERN = r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"
PHONE_PATTERN = r"\b(?:\+?\d{1,3}[\s\-]?)?\(?\d{2,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{4}\b"


def extract_regex_entities(text: str) -> list[tuple[str, str]]:

    found = []

    # Compile and search each date pattern
    for pat in DATE_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            found.append((m.group(), "DATE"))

    # Compile and search each money pattern
    for pat in MONEY_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            found.append((m.group().strip(), "MONEY"))

    # Email addresses
    for m in re.finditer(EMAIL_PATTERN, text):
        found.append((m.group(), "EMAIL"))

    # Phone numbers
    for m in re.finditer(PHONE_PATTERN, text):
        found.append((m.group(), "PHONE"))

    return found


def extract_dictionary_entities(text: str) -> list[tuple[str, str]]:
    """
    Slide a window of 1, 2, and 3 consecutive words over the text
    and check each n-gram against the gazetteer dictionaries.
    """
    found = []

    # Lowercase and remove punctuation for comparison only
    clean = text.lower().translate(str.maketrans("", "", string.punctuation))
    words = clean.split()

    # Check 3-grams, 2-grams, then 1-grams (longer match wins)
    checked_positions = set()   # avoid double-counting overlaps

    for n in (3, 2, 1):
        for i in range(len(words) - n + 1):
            if i in checked_positions:
                continue
            phrase = " ".join(words[i : i + n])

            if phrase in PERSON_LIST:
                found.append((phrase.title(), "PERSON"))
                checked_positions.update(range(i, i + n))
            elif phrase in ORGANIZATION_LIST:
                found.append((phrase.title(), "ORGANIZATION"))
                checked_positions.update(range(i, i + n))
            elif phrase in LOCATION_LIST:
                found.append((phrase.title(), "LOCATION"))
                checked_positions.update(range(i, i + n))

    return found


def ner_rule_based(text: str):
    """
    Combine regex-based extraction and dictionary-lookup extraction
    to produce a full entity list without any external NLP library.
    """
    print("\n" + "=" * 60)
    print("  PART 2A — RULE-BASED NER (Regex + Dictionary)")
    print("=" * 60)
    print(f"\nInput Text:\n  \"{text}\"\n")

    regex_entities      = extract_regex_entities(text)
    dict_entities       = extract_dictionary_entities(text)
    all_entities        = dict_entities + regex_entities

    if all_entities:
        print(f"{'Entity Text':<35} {'Label'}")
        print("-" * 55)
        for ent_text, ent_label in all_entities:
            print(f"{ent_text:<35} {ent_label}")
    else:
        print("  No entities detected.")

    return all_entities


# 2B : ML-BASED NER (CRF) 
try:
    import sklearn_crfsuite          # Wraps the fast CRFsuite C library
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False


TRAINING_DATA = [
    # Sentence 1
    [("Tim",        "B-PERSON"),
     ("Cook",       "I-PERSON"),
     ("is",         "O"),
     ("the",        "O"),
     ("CEO",        "O"),
     ("of",         "O"),
     ("Apple",      "B-ORG"),
     (".",          "O")],

    # Sentence 2
    [("Elon",       "B-PERSON"),
     ("Musk",       "I-PERSON"),
     ("founded",    "O"),
     ("Tesla",      "B-ORG"),
     ("in",         "O"),
     ("California", "B-LOC"),
     (".",          "O")],

    # Sentence 3
    [("Google",     "B-ORG"),
     ("is",         "O"),
     ("headquartered","O"),
     ("in",         "O"),
     ("Mountain",   "B-LOC"),
     ("View",       "I-LOC"),
     (".",          "O")],

    # Sentence 4
    [("Barack",     "B-PERSON"),
     ("Obama",      "I-PERSON"),
     ("visited",    "O"),
     ("India",      "B-LOC"),
     ("and",        "O"),
     ("met",        "O"),
     ("Narendra",   "B-PERSON"),
     ("Modi",       "I-PERSON"),
     (".",          "O")],

    # Sentence 5
    [("Microsoft",  "B-ORG"),
     ("acquired",   "O"),
     ("LinkedIn",   "B-ORG"),
     ("for",        "O"),
     ("$",          "O"),
     ("26",         "O"),
     ("billion",    "O"),
     (".",          "O")],

    # Sentence 6
    [("Sundar",     "B-PERSON"),
     ("Pichai",     "I-PERSON"),
     ("leads",      "O"),
     ("Google",     "B-ORG"),
     ("from",       "O"),
     ("its",        "O"),
     ("headquarters","O"),
     ("in",         "O"),
     ("New",        "B-LOC"),
     ("York",       "I-LOC"),
     (".",          "O")],

    # Sentence 7
    [("Amazon",     "B-ORG"),
     ("was",        "O"),
     ("founded",    "O"),
     ("by",         "O"),
     ("Jeff",       "B-PERSON"),
     ("Bezos",      "I-PERSON"),
     ("in",         "O"),
     ("Seattle",    "B-LOC"),
     (".",          "O")],
]


def word_features(sentence: list[str], index: int) -> dict:
  
    word = sentence[index]
    w = word.lower()

    feats = {
        "bias":         1.0,        # Always-on bias term
        "word.lower":   w,
        "word[-3:]":    word[-3:],
        "word[-2:]":    word[-2:],
        "word[:2]":     word[:2],
        "word.isupper": word.isupper(),
        "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(),
        "word.in_person_dict":  w in PERSON_LIST,
        "word.in_org_dict":     w in ORGANIZATION_LIST,
        "word.in_loc_dict":     w in LOCATION_LIST,
    }

    # ── Previous token features ──────────────────────────────────
    if index > 0:
        prev = sentence[index - 1]
        feats.update({
            "-1:word.lower":   prev.lower(),
            "-1:word.istitle": prev.istitle(),
            "-1:word.isupper": prev.isupper(),
        })
        if index > 1:
            prev2 = sentence[index - 2]
            feats["-2:word.lower"] = prev2.lower()
    else:
        feats["BOS"] = True         # Beginning Of Sentence
    
    if index < len(sentence) - 1:
        nxt = sentence[index + 1]
        feats.update({
            "+1:word.lower":   nxt.lower(),
            "+1:word.istitle": nxt.istitle(),
            "+1:word.isupper": nxt.isupper(),
        })
        if index < len(sentence) - 2:
            nxt2 = sentence[index + 2]
            feats["+2:word.lower"] = nxt2.lower()
    else:
        feats["EOS"] = True         # End Of Sentence

    return feats


def sentence_to_features(sentence: list[str]) -> list[dict]:
    """Return a list of feature dicts — one per token."""
    return [word_features(sentence, i) for i in range(len(sentence))]


def sentence_to_labels(tagged_sentence: list[tuple]) -> list[str]:
    """Extract just the BIO label strings from tagged training data."""
    return [label for _, label in tagged_sentence]


def bio_to_entities(tokens: list[str], labels: list[str]) -> list[tuple[str, str]]:
    """
    Convert a BIO-tagged sequence back into (entity_text, entity_type) pairs.

    B-PERSON → start of a PERSON entity
    I-PERSON → continuation; merge with the previous B-PERSON token
    O        → not an entity; reset the current buffer
    """
    entities = []
    current_tokens = []
    current_label  = None

    for token, label in zip(tokens, labels):
        if label.startswith("B-"):
            # Save any entity we were building
            if current_tokens:
                entities.append((" ".join(current_tokens), current_label))
            # Start a new entity
            current_tokens = [token]
            current_label  = label[2:]          # strip the "B-" prefix

        elif label.startswith("I-") and current_label == label[2:]:
            # Continuation of the same entity type
            current_tokens.append(token)

        else:
            # O tag or mismatch → flush
            if current_tokens:
                entities.append((" ".join(current_tokens), current_label))
            current_tokens = []
            current_label  = None

    # Don't forget the last entity if the sentence ended inside one
    if current_tokens:
        entities.append((" ".join(current_tokens), current_label))

    return entities


def ner_ml_based(text: str):
    print("\n" + "=" * 60)
    print("  PART 2B — ML-BASED NER (Conditional Random Field)")
    print("=" * 60)

    if not CRF_AVAILABLE:
        print("\n  [!] sklearn-crfsuite is not installed.")
        print("      Run:  pip install sklearn-crfsuite")
        return []

    # Step 1 : Prepare training data
    X_train, y_train = [], []
    for tagged_sentence in TRAINING_DATA:
        words  = [w for w, _ in tagged_sentence]
        X_train.append(sentence_to_features(words))
        y_train.append(sentence_to_labels(tagged_sentence))

    # Step 2 : Train the CRF 
    # algorithm  = 'lbfgs'  → Limited-memory BFGS (gradient method)
    # c1         → L1 regularisation (promotes sparse weights)
    # c2         → L2 regularisation (prevents overfitting)
    # max_iterations → stop after this many optimisation steps
    # all_possible_transitions → pre-build all label→label pairs
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=200,
        all_possible_transitions=True,
    )
    crf.fit(X_train, y_train)
    print("\n  CRF model trained on the sample corpus.\n")

    # Step 3 : Tokenise the input text ─────────────────────────
    clean_text = text.translate(
        str.maketrans(string.punctuation, " " * len(string.punctuation))
    )
    tokens = clean_text.split()

    if not tokens:
        print("  Input text is empty after tokenisation.")
        return []

    # Step 4 : Extract features for input tokens 
    X_test = [sentence_to_features(tokens)]

    # Step 5 : Predict BIO labels 
    predicted_labels = crf.predict(X_test)[0]

    # Step 6 : Convert BIO → entity spans 
    entities = bio_to_entities(tokens, predicted_labels)

    print(f"Input Text:\n  \"{text}\"\n")

    if entities:
        print(f"{'Token':<25} {'Predicted BIO Tag'}")
        print("-" * 45)
        for tok, lbl in zip(tokens, predicted_labels):
            marker = " <" if lbl != "O" else ""
            print(f"  {tok:<23} {lbl}{marker}")

        print(f"\n{'Entity Text':<30} {'Label'}")
        print("-" * 45)
        for ent_text, ent_label in entities:
            print(f"  {ent_text:<28} {ent_label}")
    else:
        print("  No entities detected (try adding more training data).")

    return entities


#  MAIN — Run all three approaches on sample sentences
if __name__ == "__main__":

    sample_texts = [
        "Sundar Pichai of Google met Narendra Modi in New Delhi on "
        "January 15, 2024, to discuss AI policy.",
    ]

    for i, text in enumerate(sample_texts, 1):
        print(f"\n\n{'#' * 60}")
        print(f"#  SAMPLE TEXT {i}")
        print(f"{'#' * 60}")

        # ── PART 1 : spaCy ───────────────────────────────────────
        ner_with_library(text)

        # ── PART 2A : Rule-Based ─────────────────────────────────
        ner_rule_based(text)

        # ── PART 2B : CRF (ML) ───────────────────────────────────
        ner_ml_based(text)