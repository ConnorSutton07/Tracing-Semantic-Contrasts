from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import List, Dict
import re
import numpy as np

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

def create_collocate_dict(text: List[str], node_word: str, span: int) -> Dict[str, int]:
    slice = span // 2
    frequencies = dict()
    node_indices = [i for i, word in enumerate(text) if word == node_word]
    for i in range(len(node_indices)):
        if (node_indices[i]-slice) < 0:
            collocates = text[0:(node_indices[i]+slice)]
        else:
            collocates = text[(node_indices[i] - slice):(node_indices[i] + (slice + 1))]
            collocates.pop(slice)
        for word in collocates:
            if word in frequencies: frequencies[word] += 1
            else: frequencies[word] = 1
    return frequencies

def merge_dict(dictionaries: Dict[str, Dict[str, int]], filter: bool=False) -> Dict[str, int]:
    dictionaries = list(dictionaries.items())
    merged_dict = {}
    for i in range(len(dictionaries)):
        text = dictionaries[i][1]
        merged_dict = {x: merged_dict.get(x, 0) + text.get(x, 0) for x in set(merged_dict).union(text)}
        if filter: merged_dict = {key:val for key, val in merged_dict.items() if val > 3} #Remove collocates < 3
    return merged_dict

def mi_scores(connocates: Dict[str,int], word_frequencies: Dict[str,int], node_word:str, span:int) -> Dict[str, int]:
    corpus_size = 0
    mi_scores = {}
    node_count = word_frequencies.get(node_word)

    for _, values in word_frequencies.items():
        corpus_size += values

    for collocate, near_count in connocates.items():
        collocate_count = word_frequencies.get(collocate)
        score = np.log((near_count * corpus_size)/(node_count * collocate_count * span))/np.log(2)
        mi_scores[collocate] = score
    mi_scores = {key:val for key, val in mi_scores.items() if val > 1}
    return dict(sorted(mi_scores.items(), key=lambda x: x[1], reverse=True))