import re
import string

try:
    import spacy
    spacy_nlp = spacy.load('en_core_web_sm')
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
except Exception:  # spacy or model not available
    spacy_nlp = None
    stop_words = set()

punctuations = string.punctuation


def _simple_tokenizer(sentence: str):
    sentence = re.sub("'", "", sentence)
    sentence = re.sub(r"\w*\d\w*", "", sentence)
    sentence = re.sub(" +", " ", sentence)
    sentence = re.sub(r"\n: '\'.*", "", sentence)
    sentence = re.sub(r"\n!.*", "", sentence)
    sentence = re.sub(r"^:'\'.*", "", sentence)
    sentence = re.sub(r"\n", " ", sentence)
    sentence = re.sub(r"[^\w\s]", " ", sentence)

    tokens = sentence.split()
    tokens = [t.lower() for t in tokens]
    tokens = [t for t in tokens if t not in stop_words and t not in punctuations and len(t) > 2]
    return tokens


def spacy_tokenizer(sentence: str):
    """Tokenize text using spaCy if available, otherwise fall back to a simple tokenizer."""
    sentence = re.sub("'", "", sentence)
    sentence = re.sub(r"\w*\d\w*", "", sentence)
    sentence = re.sub(" +", " ", sentence)
    sentence = re.sub(r"\n: '\'.*", "", sentence)
    sentence = re.sub(r"\n!.*", "", sentence)
    sentence = re.sub(r"^:'\'.*", "", sentence)
    sentence = re.sub(r"\n", " ", sentence)
    sentence = re.sub(r"[^\w\s]", " ", sentence)

    if spacy_nlp is None:
        return _simple_tokenizer(sentence)

    tokens = spacy_nlp(sentence)
    tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]
    tokens = [word for word in tokens if word not in stop_words and word not in punctuations and len(word) > 2]
    return tokens
