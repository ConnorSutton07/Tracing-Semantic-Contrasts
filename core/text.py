from wordcloud import WordCloud
from typing import List, Set
import os.path as osp
from core import nlp 
import json

class Text:
    __slots__ = ['__dict__', 'title', 'author', 'year', 'type', 'file', 'gender']
    def __init__(self, path: str, **kwargs):
        for k, v in kwargs.items(): 
            if k in self.__slots__: setattr(self, k, v)
        self.path = path
        self.text = self.load_text(osp.join(self.path, self.file))
    
    def get_info(self) -> None:
        return f"Author: {self.author}\nTitle:  {self.title}\nYear:   {self.year}"

    def head(self, length: int = 100):
        return self.text[:length]

    def generate_wordcloud(self, size: int, stopwords: List[str]) -> WordCloud:
        return WordCloud(
            stopwords = stopwords, 
            background_color = "black",
            colormap = 'spring',
            width = size[0], 
            height = size[1]
        ).generate(self.text)

    @staticmethod
    def load_text(file: str) -> str:
        with open(file, encoding = 'utf8') as f: contents = f.read()
        return contents

    def preprocess(self, stopwords: Set[str]) -> List[str]:
        preprocessed_text = nlp.preprocess_text(self.text, stopwords = stopwords)
        with open(osp.join(self.path, 'preprocessed', self.file), 'w', encoding = 'utf8') as f: 
            json.dump(preprocessed_text, f)

    def preprocessed_text(self, stopwords: Set[str]) -> List[str]:
        path = osp.join('preprocessed')
        if not osp.exists(osp.join(self.path, 'preprocessed', self.file)):
            self.preprocess(stopwords)
        with open(osp.join(self.path, 'preprocessed', self.file), 'r', encoding = 'utf8') as f: 
            contents = json.load(f)
        return contents 