import spacy
import re

URL_PATTERN = re.compile(r"http\S+|www\S+|https\S+", re.MULTILINE)
HTML_PATTERN = re.compile(r"<.*?>")
NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z\s]")
MULTI_SPACE_PATTERN = re.compile(r"\s+")

def load_spacy_model():
    try:
        return spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    except OSError as e:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found. "
            "Run: python -m spacy download en_core_web_sm"
        ) from e
    
nlp = load_spacy_model()

def remove_unwanted_chars(text):
    text = URL_PATTERN.sub("", text)
    text = HTML_PATTERN.sub("", text)
    text = NON_ALPHA_PATTERN.sub("", text)
    text = MULTI_SPACE_PATTERN.sub(" ", text).strip()
    return text

def lemmatize_text(text):
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and token.is_alpha
    ]
    return ' '.join(tokens)

def preprocess_text(text):
    text = text.lower()
    text = remove_unwanted_chars(text)
    text = lemmatize_text(text)
    return text

if __name__ == '__main__':
    text = 'My name is Mridul. I am 18 years old. I live in New Delhi.'
    text = preprocess_text(text)
    print(f'After preprocessing:\n{text}')