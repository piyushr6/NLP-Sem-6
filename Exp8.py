import sys
# ---- Tree rendering (shared by all three approaches) ----

def render_tree(node, prefix="", is_last=True, is_root=True):
    # Renders any tree node using box-drawing characters.
    # Accepts ParseNode (approach 1) or dict node (approach 3).
    connector = "" if is_root else ("└── " if is_last else "├── ")
    extension = "" if is_root else ("    " if is_last else "│   ")

    if isinstance(node, ParseNode):
        label = f"[{node.label}]  {node.word}" if node.is_leaf() else f"({node.label})"
        print(prefix + connector + label)
        for i, child in enumerate(node.children):
            render_tree(child, prefix + extension, i == len(node.children) - 1, False)

    elif isinstance(node, dict):
        label = f"[{node['label']}]  {node['word']}" if node.get("word") else f"({node['label']})"
        print(prefix + connector + label)
        children = node.get("children", [])
        for i, child in enumerate(children):
            render_tree(child, prefix + extension, i == len(children) - 1, False)


def render_nltk_tree(tree, prefix="", is_last=True, is_root=True):
    # Renders an NLTK Tree object using box-drawing characters.
    from nltk import Tree
    connector = "" if is_root else ("└── " if is_last else "├── ")
    extension = "" if is_root else ("    " if is_last else "│   ")

    if isinstance(tree, Tree):
        print(prefix + connector + f"({tree.label()})")
        children = list(tree)
        for i, child in enumerate(children):
            render_nltk_tree(child, prefix + extension, i == len(children) - 1, False)
    else:
        print(prefix + connector + f"[word]  {tree}")


def bracket_to_dict(s):
    # Converts a PTB bracket string like (ROOT (S (NP (DT the)) ...))
    # into a nested dict that render_tree() can consume.
    s = s.strip()
    pos = [0]

    def parse():
        while pos[0] < len(s) and s[pos[0]] == ' ':
            pos[0] += 1
        if pos[0] >= len(s):
            return None
        if s[pos[0]] == '(':
            pos[0] += 1
            label = ""
            while pos[0] < len(s) and s[pos[0]] not in (' ', ')'):
                label += s[pos[0]]
                pos[0] += 1
            children = []
            while pos[0] < len(s) and s[pos[0]] != ')':
                if s[pos[0]] == ' ':
                    pos[0] += 1
                    continue
                child = parse()
                if child:
                    children.append(child)
            pos[0] += 1
            # if only child is a bare word, treat this node as a leaf
            if (len(children) == 1
                    and not children[0].get("children")
                    and children[0].get("word") is None):
                return {"label": label, "word": children[0]["label"], "children": []}
            return {"label": label, "children": children, "word": None}
        else:
            word = ""
            while pos[0] < len(s) and s[pos[0]] not in (' ', ')'):
                word += s[pos[0]]
                pos[0] += 1
            return {"label": word, "children": [], "word": None}

    return parse()


def render_bracket_tree(bracket_str):
    node = bracket_to_dict(bracket_str)
    if node:
        render_tree(node)
    else:
        print("  (could not render tree)")


# ---- Approach 1: Manual CFG parser (no libraries)
class ParseNode:
    def __init__(self, label, children=None, word=None):
        self.label = label
        self.children = children or []
        self.word = word

    def is_leaf(self):
        return self.word is not None

    def to_bracket(self):
        if self.is_leaf():
            return f"({self.label} {self.word})"
        return f"({self.label} {' '.join(c.to_bracket() for c in self.children)})"


class ManualCFGParser:
    # Lexicon: maps POS tag -> set of accepted words
    LEXICON = {
        "Det":   {"the", "a", "an"},
        "N":     {"dog", "cat", "man", "woman", "park", "bone", "ball", "garden"},
        "Adj":   {"big", "small", "old", "happy", "brown"},
        "PropN": {"John", "Mary", "Alice", "Bob"},
        "PRP":   {"he", "she", "they", "i"},
        "V":     {"saw", "chased", "runs", "ate", "found", "played", "likes"},
        "P":     {"in", "on", "with", "near", "at"},
    }

    # Phrase structure rules
    PHRASE_RULES = {
        "S":  [["NP", "VP"]],
        "NP": [["Det", "Adj", "N"], ["Det", "N"], ["PropN"], ["PRP"]],
        "VP": [["V", "NP", "PP"], ["V", "NP"], ["V", "PP"], ["V"]],
        "PP": [["P", "NP"]],
    }

    def _pos_tag(self, word):
        return [tag for tag, words in self.LEXICON.items() if word.lower() in words]

    def _parse_terminal(self, tokens, pos, tag):
        if pos < len(tokens) and tag in self._pos_tag(tokens[pos]):
            return ParseNode(tag, word=tokens[pos]), pos + 1
        return None, pos

    def _parse_non_terminal(self, tokens, pos, symbol):
        if symbol in self.PHRASE_RULES:
            for production in self.PHRASE_RULES[symbol]:
                node, new_pos = self._try_production(tokens, pos, symbol, production)
                if node is not None:
                    return node, new_pos
            return None, pos
        return self._parse_terminal(tokens, pos, symbol)

    def _try_production(self, tokens, pos, lhs, rhs):
        children, cur_pos = [], pos
        for sym in rhs:
            child, cur_pos = self._parse_non_terminal(tokens, cur_pos, sym)
            if child is None:
                return None, pos
            children.append(child)
        return ParseNode(lhs, children=children), cur_pos

    def parse(self, sentence):
        tokens = sentence.strip().split()
        tree, end_pos = self._parse_non_terminal(tokens, 0, "S")
        return tree if (tree and end_pos == len(tokens)) else None


# ---- Approach 2: NLTK CFG ChartParser ----

def nltk_cfg_parse(sentence):
    try:
        import nltk
        from nltk import CFG, ChartParser

        # Grammar defined as a string; each rule on its own line
        grammar_str = (
            "S -> NP VP\n"
            "NP -> Det Adj N\n"
            "NP -> Det N\n"
            "NP -> PropN\n"
            "NP -> PRP\n"
            "VP -> V NP PP\n"
            "VP -> V NP\n"
            "VP -> V PP\n"
            "VP -> V\n"
            "PP -> P NP\n"
            "Det -> 'the' | 'a' | 'an'\n"
            "N -> 'dog' | 'cat' | 'man' | 'woman' | 'park' | 'bone' | 'ball' | 'garden'\n"
            "Adj -> 'big' | 'small' | 'old' | 'happy' | 'brown'\n"
            "PropN -> 'John' | 'Mary' | 'Alice' | 'Bob'\n"
            "PRP -> 'he' | 'she' | 'they' | 'i'\n"
            "V -> 'saw' | 'chased' | 'runs' | 'ate' | 'found' | 'played' | 'likes'\n"
            "P -> 'in' | 'on' | 'with' | 'near' | 'at'\n"
        )
        grammar = CFG.fromstring(grammar_str)
        tokens = sentence.lower().strip().split()
        parser = ChartParser(grammar)
        return list(parser.parse(tokens))

    except ImportError:
        print("NLTK not installed. Run: pip install nltk")
        return []
    except Exception as e:
        print(f"NLTK parse error: {e}")
        return []


# ---- Approach 3: Stanza neural constituency parser ----

def neural_parse_stanza(sentence):
    try:
        import stanza
        print("Loading Stanza neural pipeline...")
        nlp = stanza.Pipeline(
            lang='en',
            processors='tokenize,pos,constituency',
            verbose=False
        )
        doc = nlp(sentence)
        return [str(sent.constituency) for sent in doc.sentences]

    except ImportError:
        print("Stanza not installed. Run: pip install stanza")
        print("Then: python -c \"import stanza; stanza.download('en')\"")
        return []
    except Exception as e:
        print(f"Stanza error: {e}")
        return []


# ---- Main experiment runner ----

def run_experiment(sentence):
    print(f"\nSentence: \"{sentence}\"\n")

    # Approach 1
    print("Approach 1: Manual CFG Parser (No Libraries)")
    print()
    print("  Grammar Productions:")
    print("    S   ->  NP VP")
    print("    NP  ->  Det Adj N  |  Det N  |  PropN  |  PRP")
    print("    VP  ->  V NP PP   |  V NP   |  V PP   |  V")
    print("    PP  ->  P NP")
    print()

    parser = ManualCFGParser()
    tree1 = parser.parse(sentence)
    if tree1:
        print("  Parse Tree:")
        print()
        render_tree(tree1)
        print(f"\n  Bracket: {tree1.to_bracket()}")
    else:
        print("  Could not parse - sentence may use words outside the grammar vocabulary.")
        print("  Try: 'the big dog chased a cat in the park'")

    # Approach 2
    print("\nApproach 2: NLTK CFG ChartParser (with Library)")
    print()

    nltk_trees = nltk_cfg_parse(sentence)
    if nltk_trees:
        print(f"  Found {len(nltk_trees)} parse tree(s).")
        for i, t in enumerate(nltk_trees[:3], 1):
            print(f"\n  Tree #{i}:")
            print()
            render_nltk_tree(t)
            print(f"\n  Bracket: {t}")
    else:
        print("  No parse trees found.")

    # Approach 3
    print("\nApproach 3: Stanza Neural Constituency Parser")
    print()

    stanza_trees = neural_parse_stanza(sentence)
    if stanza_trees:
        for raw in stanza_trees:
            print("  Parse Tree:")
            print()
            render_bracket_tree(raw)
            print(f"\n  Bracket: {raw}")
    else:
        print("  No neural parse available.")

    print("\nDone.")


if __name__ == "__main__":
    DEFAULT = "the big dog chased a cat in the park"
    if len(sys.argv) > 1:
        sentence = " ".join(sys.argv[1:])
    else:
        sentence = DEFAULT
        print(f"No input given. Using default: \"{DEFAULT}\"")
        print("Usage: python Exp8.py <your sentence here>")
    run_experiment(sentence)