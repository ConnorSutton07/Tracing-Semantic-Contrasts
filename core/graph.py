from __future__ import annotations
from adjustText import adjust_text
import matplotlib.pyplot as plt
import numpy as np

def scatter_embeddings(label: str, words: List[str], pcs, explained_variance, save_path: str, adjust_annotations: bool = True) -> None:
    with plt.style.context('Solarize_Light2'):
        plt.figure(figsize=(12, 8))
        plt.rcParams.update({'font.family':'serif'})
        plt.scatter(pcs[:, 0], pcs[:, 1], c='darkgoldenrod')

        annotations = []
        for word, x, y in zip(words, pcs[:, 0], pcs[:, 1]):
            annotations.append(plt.annotate(word, xy=(x+0.015, y-0.005), xytext=(0, 0), textcoords='offset points'))
        if adjust_annotations:
            adjust_text(annotations, lim=10)

        plt.xlabel("PC1 | " + "{:.2%}".format(explained_variance[0]))
        plt.ylabel("PC2 | " + "{:.2%}".format(explained_variance[1]))
        plt.title(f"Word Embeddings | Translation: {label}") 
        plt.savefig(save_path, dpi=200)

def tabulate_embeddings(embeddings: dict, save_path: str, keyword_set: str, length: int):
    with plt.style.context('Solarize_Light2'):
        fig = plt.figure(figsize=(12, 3))
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 12
        })
        rowColours = ['goldenrod']
        rowLabels = [f"{i}" for i in range(1, length + 1)]
        values = np.array(list(embeddings.values())).T
        headers = list(embeddings.keys())
        tbl = plt.table(cellText = values, rowLabels = rowLabels, colLabels = headers, loc = "center")
        tbl.scale(1, 2)
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        plt.title(f"Closest Semantic Neighbors For Each Keyword | {keyword_set}")
        plt.axis('off')
        plt.savefig(save_path, dpi=120)