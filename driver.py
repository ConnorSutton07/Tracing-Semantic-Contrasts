import os
import json
from core import ui, settings, nlp 
from core.text import Text
from typing import List, Dict, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt 
import multiprocessing as mp 
import numpy as np
import time 

class Driver:
    def __init__(self) -> None:
        self.paths = {}
        self.paths["current"]   = os.getcwd()
        self.paths["corpora"]   = os.path.join(self.paths["current"], "corpora")
        self.paths["figures"]   = os.path.join(self.paths["current"], "figures")
        self.paths["data"]    = os.path.join(self.paths["current"], "data")

        # list of methods that user will be able to choose from 
        self.modes = [
            ("Compare Across Corpora", self.inter_corpus_analysis),
            ("Compare Within Corpus",  self.intra_corpus_analysis),
            ("Generate Wordclouds",    self.generate_wordclouds),
            ("Generate data",     self.generate_data),
            ("Print Corpus Contents",  self.print_corpus_info),
            ("Plot Documents by Decade", self.plot_documents_by_decade),
            ("Generate Mutual Information Scores", self.generate_MIscores)
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
        """
        Compare corpus based on data such as author, gender, etc.
        """
        corpus, name = self.select_corpus()
        name1, name2, fun1, fun2 = self.select_analysis_topics()
        part1 = [txt for txt in corpus if fun1(txt)]
        part2 = [txt for txt in corpus if fun2(txt)]
        query = input('Word: ')
        
        self.generate_intra_MIscores(name, corpus, name1, part1, query)
        self.generate_intra_MIscores(name, corpus, name2, part2, query)



    def generate_data(self):
        """
        Preprocesses text and saves a frequency dictionary to the
        data directory

        In the future, this could also generate the word embeddings

        """
        corpus, name = self.select_corpus()
        if corpus is None: return

        print("Creating word frequency dictionary...")
        frequency_dict = dict()

        for document in tqdm(corpus):
            text = document.preprocessed_text() #nlp.preprocess_text(document.text, stopwords = settings.stopwords)
            cur_dict = nlp.create_frequency_dict(text)
            if '' in cur_dict: del cur_dict['']
            frequency_dict[document.title] = cur_dict

        path = os.path.join(self.get_path([self.paths["data"], name]), "frequencies.json")
        ui.saveToJSON(frequency_dict, path)

        #preprocessed_text = nlp.preprocess_text(corpus)

    def generate_collocates(self, query: str, corpus: List[Text], name: str):
        """
        Create collocate dictionary with a given string.

        """
        collocate_dict = {}
        args = list(zip(corpus, [query] * len(corpus)))

        data = []
        with mp.Pool(processes = 4) as p:
            data.append(list(tqdm(p.imap_unordered(
                self.generate_collocate_work, args), 
                desc = "Creating collocate dictionaries", 
                total = len(args)))
            )
        data = data[0]
        titles = [x[0] for x in data if x[1]]
        collocates = [x[1] for x in data if x[1]]
        collocate_dict = dict(zip(titles, collocates))

        path = os.path.join(self.get_path([self.paths["data"], name]), f"{query}_collocates.json")
        ui.saveToJSON(collocate_dict, path)

    def generate_MIscores(self):
        """
        Calculate MI score given a string. This could be iterative.
        """
        corpus, name = self.select_corpus()
        if corpus is None: return
        query = input("Word: ")

        frequency_path = self.get_path([self.paths['data'], name, 'frequencies.json'])
        infile = open(frequency_path)
        frequency_file = json.load(infile)
        model_path = os.path.join(self.get_path([self.paths["data"], name]), f"{query}_collocates.json")

        if not os.path.isfile(model_path): self.generate_collocates(query, corpus, name)

        with open(model_path, 'r') as f:
            collocate_file = json.load(f)

        collocates = nlp.merge_dict(collocate_file, True)
        frequencies = nlp.merge_dict(frequency_file)
        mi_scores = nlp.mi_scores(collocates, frequencies, query, 4)
        save_path = os.path.join(self.get_path([self.paths["data"], name]), f"{query}_MIscores.json")
        ui.saveToJSON(dict({query : mi_scores}), save_path)

    def generate_intra_MIscores(self, name, corpus, part_name, part, query):
        """
        Calculate the MI score of a given partition of a corpus
        """

        frequency_path = self.get_path([self.paths['data'], name, 'frequencies.json'])
        infile = open(frequency_path)
        frequency_file = json.load(infile)
        frequency_file = {txt.title:frequency_file[txt.title] for txt in part} 
        model_path = os.path.join(self.get_path([self.paths["data"], name]), f"{query}_collocates.json")

        if not os.path.isfile(model_path): self.generate_collocates(query, corpus, name)

        with open(model_path, 'r') as f:
            collocate_file = json.load(f)

        collocate_file = {txt.title:collocate_file[txt.title] for txt in part if txt.title in
                collocate_file.keys()}
        collocates = nlp.merge_dict(collocate_file, True)
        frequencies = nlp.merge_dict(frequency_file)
        mi_scores = nlp.mi_scores(collocates, frequencies, query, 4)
        save_path = os.path.join(self.get_path([self.paths["data"], name, part_name]), f"{query}_MIscores.json")
        ui.saveToJSON(dict({query : mi_scores}), save_path)



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

    def generate_collocate_work(self, args: Tuple[Text, str]):
        document, query = args
        text = document.preprocessed_text() # nlp.preprocess_text(document.text, stopwords=settings.stopwords)
        text = [word for word in text if word != " " and word != ""]
        cur_dict = nlp.create_collocate_dict(text, query, 6)
        #collocate_dict[document.title] = cur_dict
        return document.title, cur_dict



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

    def plot_documents_by_decade(self):
        corpus, name = self.select_corpus()
        if corpus is None: return

        years = [doc.year for doc in corpus]
        minimum = round(min(years)) / 10  # round to nearest decade
        maximum = (round(min(years)) / 10 ) + 10
        bins = (maximum - minimum) / 10

        with plt.style.context('Solarize_Light2'):
            fig = plt.figure(figsize = (12, 8))
            plt.rcParams.update({
                'font.family': 'serif',
                'font.size': 12
            })
            plt.hist(years, bins=range(1720, 1900, 10), align='mid', color='goldenrod')
            plt.title(f"Documents in {name} Corpus by Decade")
            plt.xlabel("Year")
            plt.ylabel("Number of Documents")
            plt.show()
            plt.close()

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

    def select_analysis_topics(self):
        """
        Selects what to analyze a corpus based on
        """
        parts = ["Gender", "Year"]
        gens = ['Generate MI Scores']
        print('How should the corpus be partitioned:')
        for i, part in enumerate(parts, start = 1):
            print(f'\t{i}) {part}')
        part = ui.getValidInput('', dtype = int, valid=range(1, len(parts) + 1)) - 1
        if part == 0:
            name1 = 'men'
            name2 = 'women'
            fun1 = lambda txt : txt.gender == 'M'
            fun2 = lambda txt : txt.gender == 'F'
        elif part == 1:
            range1, range2 = [0,0], [0,0]
            range1[0] = ui.getValidInput('Oldest year in first range', dtype = int,
                valid=range(1700,1900))
            range1[1] = ui.getValidInput('Newest year in first range', dtype = int,
                valid=range(1700,1900))
            range2[0] = ui.getValidInput('Oldest year in second range', dtype = int,
                valid=range(1700,1900))
            range2[1] = ui.getValidInput('Newest year in second range', dtype = int,
                valid=range(1700,1900))
            name1 = f'{range1[0]}-{range1[1]}'
            name2 = f'{range2[0]}-{range2[1]}'
            fun1 = lambda txt : txt.year >= range1[0] and txt.year <= range1[1]
            fun2 = lambda txt : txt.year >= range2[0] and txt.year <= range2[1]
        '''
        print('What should be generated:')
        for i, gen in enumerate(gens, start = 1):
            print(f'\t{i}) {gen}')
        part = ui.getValidInput('> ', dtype = int, valid=range(1, len(gens) + 1)) - 1
        '''     
        return name1, name2, fun1, fun2
            
