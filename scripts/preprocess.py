import spacy
import re

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def remove_unwanted_chars(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text 

def lemmatize_text(text):
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop
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