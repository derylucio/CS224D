import sys, csv, collections
import numpy as np

#will need to unpack the information.
class DataProcessor:

  def __init__(self):
    self.word_index_map = {"PADDING" : 0}
    #let the zero index be for the padding
    self.num_unique = 1
    self.train_file = ".../../data/train_small.tsv"
    self.test_file = ".../.../data/test_small.tsv"
    self.valid_file = ".../.../data/valid_small.tsv"
    self.ESSAY_INDEX = 2
    self.ESSAY_ID_INDEX = 0
    self.DOMAIN_1_INDEX = 6


  def readInData(self, filename):
  	with open(filename, 'rb') as tsv:
  		data = csv.reader(tsv, delimiter='\n')
      start = True
      iterdata = iter(data)
  		for row in iterdata:
        wv_indices = np.zeros(550)
        scores = np.zeros(1)
  			entries = row.split("\t")
        print entries
        essay = entries[self.ESSAY_INDEX]
        scores[0] = entries[self.DOMAIN_1_INDEX]
        words = essay.split()
        i = 0
        for word in words:
          if word in self.word_index_map:
            wv_indices[i] = self.word_index_map[word]
          else:
            self.word_index_map[word] = self.num_unique
            wv_indices[i] = self.num_unique
            self.num_unique += 1
          i += 1
        if start:
          essay_list = wv_indices
          essay_scores = scores
          start = False
        else:
          essay_scores = np.vstack((essay_scores, scores))
          essay_list = np.vstack((essay_list, wv_indices))
    return essay_list, essay_scores

  def getData(self, dataset):
    if dataset == 0:
      essays, scores = self.readInData(self.train_file)
    elif dataset == 1:
      essays, scores = self.readInData(self.test_file)
    else:
      essays, scores = self.readInData(self.valid_file)
    return essays, scores

  def getNumUnique(self):
    return self.num_unique