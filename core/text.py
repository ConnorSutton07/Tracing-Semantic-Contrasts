from wordcloud import WordCloud
from typing import List
import os

class Text:
    __slots__ = ['__dict__', 'title', 'author', 'year', 'type', 'file', 'gender']
    def __init__(self, path: str, **kwargs):
        for k, v in kwargs.items(): 
            if k in self.__slots__: setattr(self, k, v)
        self.text = self.load_text(os.path.join(path, self.file))
    
    def get_info(self) -> None:
        return f"Author: {self.author}\nTitle:  {self.title}\nYear:   {self.year}"

    def head(self, length: int = 25):
        return self.text[:length]

    def generate_wordcloud(self, size: int, stopwords: List[str]) -> WordCloud:
        return #return ()

    @staticmethod
    def load_text(file: str) -> str:
        with open(file, encoding = 'utf8') as f: contents = f.read()
        return f