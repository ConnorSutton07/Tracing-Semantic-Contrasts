from nltk import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as stopwords_model
from functools import lru_cache
from typing import List, Dict, Set
from gensim.models.fasttext import FastText
from sklearn.decomposition import PCA
from unidecode import unidecode
import re
import numpy as np


def tokenize(text: str) -> List[List[str]]:
    sentences = sent_tokenize(text)
    tokens = [word_tokenize(sent) for sent in sentences]
    return tokens

@lru_cache(maxsize=2**16)  # 2**15 = 32768
def lemmatize(word: str) -> str:
    return WordNetLemmatizer().lemmatize(word)

def filter_text(text: str) -> str:
    REMOVE_PTN = re.compile(r'[^a-z]+')
    return re.sub(REMOVE_PTN, lambda m: ' ', unidecode(text).lower())

def valid_token(token: str) -> bool:
    return token not in STOPWORDS and len(token) > 1

def preprocess_text(text: str, stopwords: Set[str]) -> List[str]:
    """ 
    Prepares text to be analyzed
    Tokenizes, lemmatizes, and removes stopwords

    """
    filtered = filter_text(text)
    tokens = tokenize(filtered)
    cleaned = [[w for w in sent if w not in stopwords] for sent in tokens]
    # cleaned = [w for w in tokens if token not in stopwords]
    # cleaned = [[w for w in sent if token not in stopwords] for sent in tokens]
    #lemmatized = [lemmatize(w) for w in cleaned]
    lemmatized = [[lemmatize(w) for w in sent] for sent in cleaned]
    return [[w for w in sent if len(w) > 1] for sent in lemmatized]

def create_frequency_dict(text: List[str]) -> Dict[str, int]:
    full_text = []
    for sent in text:
        full_text = full_text + sent 
    
    frequencies = dict()
    for word in full_text:
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
        mi_scores[collocate] = np.round(score, 2)
    mi_scores = {key:val for key, val in mi_scores.items() if val > 2.5}
    return dict(sorted(mi_scores.items(), key=lambda x: x[1], reverse=True))

def word_embeddings(key_words: List[str], kwargs: dict, model = None, text: List[str] = None):
    if model is None: model = FastText(text, **kwargs)

    words = {words: [item[0] for item in model.wv.most_similar([words], topn = 100)] for words in key_words}
    flattened_words = np.array(sum([[k] + v for k, v in words.items()], []))
    vectors = model.wv[flattened_words]
    reduced_model = PCA(n_components = 2)
    pcs = reduced_model.fit_transform(vectors)
    variance_ratio = reduced_model.explained_variance_ratio_

    return model, words, pcs, variance_ratio