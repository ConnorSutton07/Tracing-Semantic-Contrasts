import os
import json
from core import ui
from core.text import Text
from typing import List, Dict

class Driver:
    def __init__(self) -> None:
        self.paths = {}
        self.paths["current"]   = os.getcwd()
        self.paths["corpora"]   = os.path.join(self.paths["current"], "corpora")
        self.paths["figures"]   = os.path.join(self.paths["current"], "figures")

        # list of methods that user will be able to choose from 
        self.modes = [
            ("Compare Across Corpora", self.inter_corpus_analysis),
            ("Compare Within Corpus", self.intra_corpus_analysis),
            ("Generate Wordclouds", self.generate_wordclouds),
            ("Print Corpus Information", self.print_corpus_info)
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

    def generate_wordclouds(self):
        raise NotImplementedError


    # ----------------------------------------------
    #              Utility Methods
    #-----------------------------------------------

    def print_corpus_info(self):
        """
        Prints the author, year, and title for each work in a corpus

        """
        corpus = self.select_corpus()
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

    # ----------------------------------------------
    #                UI Methods
    #-----------------------------------------------

    def select_mode(self):
        """
        Choose type of analysis
        Options:
            1) Compare Across Corpora (eg. Greek vs. Rennaissance)
            2) Compare Within Corpus (eg. Male vs Female Romanticists)
            3) Generate Wordclouds
            4) Print Corpus Information
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

    def select_corpus(self) -> List[Text]: 
        """
        Retrieves the 'info.json' file of the chosen corpus

        Calls retrieve_texts() to return a list of Text objects associated
        with the corpus

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
            return self.retrieve_texts(info, corpus_path) 
        return None # user has chosen to go back