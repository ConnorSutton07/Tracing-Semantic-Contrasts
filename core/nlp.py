from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import List, Dict
import re

def tokenize(text: str) -> List[str]:
    return word_tokenize(text)

def lemmatize(word: str) -> str:
    return WordNetLemmatizer().lemmatize(word)

def preprocess_text(text: str, stopwords = [], replacements: List[tuple] = None, no_lemmatization: List[str] = []) -> List[str]:
    """ 
    Prepares text to be analyzed
    Removes stopwords and unnecessary characters
    Optionally lemmatizes words (ideal for creating word embeddings)

    """

    tokens = tokenize(text)
    tokens = [t for t in tokens if t not in stopwords]

    for i, token in enumerate(tokens):
        token = re.sub(r'\W', ' ', str(token)) # remove all the special characters
        token = re.sub(r'\s+[a-zA-Z]\s+', ' ', token) # remove all single characters
        token = re.sub(r'\^[a-zA-Z]\s+', ' ', token) # Remove single characters from the start
        token = re.sub(r'[^\w\d\s]', '', token)
        token = re.sub(r'\s+', ' ', token, flags=re.I) # Substituting multiple spaces with single space
        words = token.split(' ')
        if len(words) > 1:
            for w in words[1:]: tokens.append(w)
            token = words[0]
        token = token.lower()
        token = lemmatize(token)
        tokens[i] = token

    return tokens

def create_frequency_dict(text: List[str]) -> Dict[str, int]:
    frequencies = dict()
    for word in text:
        if word in frequencies: frequencies[word] += 1
        else: frequencies[word] = 1
    return frequencies
