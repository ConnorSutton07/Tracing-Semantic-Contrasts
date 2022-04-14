from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as stopwords_model
from functools import lru_cache
from typing import List, Dict
import re
from unidecode import unidecode
import numpy as np

STOPWORDS = set(stopwords_model.words('english'))
REMOVE_PTN = re.compile(r'[^a-z]+')

def tokenize(text: str) -> List[str]:
    return word_tokenize(text)

@lru_cache(maxsize=2**16)  # 2**15 = 32768
def lemmatize(word: str) -> str:
    return WordNetLemmatizer().lemmatize(word)

def filter_text(text: str) -> str:
    return re.sub(REMOVE_PTN, lambda m: ' ', unidecode(text).lower())

def valid_token(token: str) -> bool:
    return token not in STOPWORDS and len(token) > 1

def preprocess_text(text: str) -> List[str]:
    """ 
    Prepares text to be analyzed
    Tokenizes, lemmatizes, and removes stopwords

    """
    return [lemmatize(token) for token in tokenize(filter_text(text)) if valid_token(token)]


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

def merge_dict(dictionaries: Dict[str, Dict[str, int]], filter: bool = False) -> Dict[str, int]:
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
        score = np.log((near_count * corpus_size)/(node_count * collocate_count * span)) / np.log(2)
        mi_scores[collocate] = score
    mi_scores = {key:val for key, val in mi_scores.items() if val > 1}
    return dict(sorted(mi_scores.items(), key=lambda x: x[1], reverse=True))

