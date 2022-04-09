import os
import json
from core import ui, settings, nlp 
from core.text import Text
from typing import List, Dict, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt 
import multiprocessing as mp 
import time 

class Driver:
    def __init__(self) -> None:
        self.paths = {}
        self.paths["current"]   = os.getcwd()
        self.paths["corpora"]   = os.path.join(self.paths["current"], "corpora")
        self.paths["figures"]   = os.path.join(self.paths["current"], "figures")
        self.paths["models"]    = os.path.join(self.paths["current"], "models")

        # list of methods that user will be able to choose from 
        self.modes = [
            ("Compare Across Corpora", self.inter_corpus_analysis),
            ("Compare Within Corpus",  self.intra_corpus_analysis),
            ("Generate Wordclouds",    self.generate_wordclouds),
            ("Generate Models",     self.generate_models),
            ("Print Corpus Contents",  self.print_corpus_info)
        ]

        self.introduction()

    def introduction(self) -> None:
        print()
        print("     +" + "-"*28 + "+")
        print("       TRACING SEMANTIC CONTRASTS")
        print("     +" + "-"*28 + "+")
        print()

    def run(self) -> None:
        while True:
            mode = self.select_mode()
            if mode is None: break
            mode()
        return


    # ----------------------------------------------
    #              Analysis Methods
    #-----------------------------------------------

    def inter_corpus_analysis(self):
        raise NotImplementedError 

    def intra_corpus_analysis(self):
        raise NotImplementedError

    def generate_models(self):
        """
        Preprocesses text and saves a frequency dictionary to the
        models directory

        In the future, this could also generate the word embeddings

        """
        corpus, name = self.select_corpus()
        if corpus is None: return

        print("Creating word frequency dictionary...")
        frequency_dict = dict()

        for document in tqdm(corpus):
            text = nlp.preprocess_text(document.text, stopwords = settings.stopwords)
            cur_dict = nlp.create_frequency_dict(text)
            if '' in cur_dict: del cur_dict['']
            frequency_dict[document.title] = cur_dict

        path = os.path.join(self.get_path([self.paths["models"], name]), "frequencies.json")
        ui.saveToJSON(frequency_dict, path)

            

        #preprocessed_text = nlp.preprocess_text(corpus)

    def generate_wordclouds(self):
        """
        Generates wordclouds for each text in a given corpus
        Saves to Figures/<corpus>/wordclouds/
        Multiprocessed

        """
        corpus, name = self.select_corpus()
        if corpus is None: return 

        save_path = self.get_path([self.paths['figures'], name, 'wordclouds'])
        paths = [os.path.join(save_path, text.title + ".png") for text in corpus]
        args = list(zip(corpus, paths)) 

        print("Generating wordclouds...")
        with mp.Pool(processes = 4) as p:
            list(tqdm(p.imap_unordered(self.generate_wordcloud_work, args), total = len(args)))


    # ----------------------------------------------
    #              Utility Methods
    #-----------------------------------------------

    def generate_wordcloud_work(self, args: Tuple[Text, str]):
        """
        Helper function called by generate_wordclouds
        Plots the wordcloud image with matplotlib and saves
        
        """
        text, path = args
        fig = plt.figure()
        plt.imshow(text.generate_wordcloud(
            stopwords = settings.stopwords,
            size = (600, 600)),
            interpolation = 'bilinear',
            cmap = 'Paired'
        )
        plt.axis('off')
        plt.savefig(path)
        plt.close(fig)

    def print_corpus_info(self):
        """
        Prints the author, year, and title for each work in a corpus

        """
        corpus, name = self.select_corpus()
        if corpus is None: return 

        print(f"\n{name} Texts: ")
        for text in corpus:
            print('-' * 50)
            print(text.get_info())
        print('\n\n')

    def retrieve_texts(self, info: Dict[str, any], path: str) -> List[Text]:
        """
        Reads an 'info.json' file and returns a list of Text objects

        """
        texts = []
        for entry in info: # iterate through all the texts contained in the 'info' dictionary
            texts.append(Text(path, **entry))
        texts.sort(key = lambda t: t.year) # sort texts by year (ascending)
        return texts

    def get_path(self, path_list: List[str]) -> str: 
        """
        Checks if the path from the beginning to end of the list exists
        If so, return that path
        If not, create that path and then return it

        Used for saving figures (e.g., wordclouds, graphs, etc.)

        """
        path = path_list[0]
        for i in range(1, len(path_list)):
            new_path = os.path.join(path, path_list[i])
            if not os.path.exists(new_path): os.mkdir(new_path)
            path = new_path 
        return path 

    # ----------------------------------------------
    #                UI Methods
    #-----------------------------------------------

    def select_mode(self):
        """
        Choose type of analysis
        Options:
            1) Compare Across Corpora (e.g., Greek vs. Rennaissance)
            2) Compare Within Corpus (e.g., Male vs Female Romanticists)
            3) Generate Wordclouds
            4) Print Corpus Contents
            5) Exit

        """
        num_modes = len(self.modes)
        msg = "Analysis Method:\n" 
        for i, mode in enumerate(self.modes, start = 1): # list the modes
            msg += f"   {i}) {mode[0]}\n"
        msg += f"   {num_modes + 1}) Exit" 
        choice = ui.getValidInput(msg, dtype = int, valid=range(1, num_modes + 2)) - 1 # let user choose mode
        if (choice != num_modes):
            return self.modes[choice][1] # return the method associated with the chosen mode
        return None # user has chosen to leave the program

    def select_corpus(self) -> (List[Text], str): 
        """
        Returns a list of texts for a chosen corpus as well as the corpus' name

        """
        corpora: List[str] = os.listdir(path = self.paths['corpora']) # retrieve list of subdirectories in corpora directory
        num_corpora = len(corpora)
        msg = "Select Corpus:\n"
        for i, corpus in enumerate(corpora, start = 1): # list corpora for user to choose from
            msg += f"   {i}) {corpus}\n"
        msg += f"   {num_corpora + 1}) Back"
        choice = ui.getValidInput(msg, dtype = int, valid = range(1, num_corpora + 2)) - 1
        if choice != num_corpora:
            corpus_path = os.path.join(self.paths['corpora'], corpora[choice]) # find info.json
            # open info.json and read its contents into 'info'
            with open(os.path.join(corpus_path, 'info.json')) as infile: info = json.load(infile) 
            return self.retrieve_texts(info, corpus_path), corpora[choice]
        return None # user has chosen to go back


            
