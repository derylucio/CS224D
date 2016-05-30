import sys, csv, collections, os
import numpy as np

#dummies
# self.saved_data_essay = "/Users/luciodery/Desktop/Stanford!/spring2015/CS224D/FinalProject/code/basic_nn/saved_data/checktrainessays"
# self.saved_data_score = "/Users/luciodery/Desktop/Stanford!/spring2015/CS224D/FinalProject/code/basic_nn/saved_data/checktrainscores"
# self.saved_data_essay_final = "/Users/luciodery/Desktop/Stanford!/spring2015/CS224D/FinalProject/code/basic_nn/saved_data/trinesys.npy"
# self.saved_data_score_final = "/Users/luciodery/Desktop/Stanford!/spring2015/CS224D/FinalProject/code/basic_nn/saved_data/trinsors.npy"

#will need to unpack the information.
class DataProcessor:

  def __init__(self):
    self.word_index_map = {"-1PADDING" : 0}
    #let the zero index be for the padding
    self.num_unique = 1
    self.train_file = "/Users/luciodery/Desktop/Stanford!/spring2015/CS224D/FinalProject/data/training_set.tsv"
    self.test_file = "/Users/luciodery/Desktop/Stanford!/spring2015/CS224D/FinalProject/data/test_set.tsv"
    self.valid_file = "/Users/luciodery/Desktop/Stanford!/spring2015/CS224D/FinalProject/data/valid_set.tsv"
    self.saved_data_essay = "/Users/luciodery/Desktop/Stanford!/spring2015/CS224D/FinalProject/code/basic_nn/saved_data/trainessays"
    self.saved_data_score = "/Users/luciodery/Desktop/Stanford!/spring2015/CS224D/FinalProject/code/basic_nn/saved_data/trainscores"
    self.saved_data_score_nu = "/Users/luciodery/Desktop/Stanford!/spring2015/CS224D/FinalProject/code/basic_nn/saved_data/nu"
    self.saved_data_essay_final = "/Users/luciodery/Desktop/Stanford!/spring2015/CS224D/FinalProject/code/basic_nn/saved_data/trainessays.npy"
    self.saved_data_score_final = "/Users/luciodery/Desktop/Stanford!/spring2015/CS224D/FinalProject/code/basic_nn/saved_data/trainscores.npy"
    self.saved_data_score_nu_final = "/Users/luciodery/Desktop/Stanford!/spring2015/CS224D/FinalProject/code/basic_nn/saved_data/nu.npy"
    self.ESSAY_INDEX = 2
    self.ESSAY_ID_INDEX = 0
    self.DOMAIN_1_INDEX = 6
    self.TYPE_INDEX = 1


  def readInData(self, filename, is_test):
    max_words = 0
    num_files = 0
    with open(filename, 'rb') as tsv:
      data = csv.reader(tsv, delimiter='\n')
      start = True
      iterdata = iter(data)
      next(iterdata) 
      for row in iterdata:
        num_files += 1
        row = row[0]
        wv_indices = np.zeros(950)
        if(not is_test):
          scores = np.zeros(1)
        row.replace(" ", "")
        entries = row.split("\t")
        essay = entries[self.ESSAY_INDEX]
        essay_type = int(entries[self.TYPE_INDEX])
        if  essay_type == 7 or essay_type == 8:
          continue
        if(not is_test):
          scores[0] = entries[self.DOMAIN_1_INDEX]
        words = essay.split()
        i = 0
        for word in words:
          if len(word) < 2 or (not word.isalpha()):
            continue
          if word in self.word_index_map:
            wv_indices[i] = self.word_index_map[word]
          else:
            self.word_index_map[word] = self.num_unique
            wv_indices[i] = self.num_unique
            self.num_unique += 1
          i += 1
        max_words = max_words if max_words > i else i
        if start:
          essay_list = wv_indices
          if(not is_test):
            essay_scores = scores
          start = False
        else:
          if(not is_test):
            essay_scores = np.vstack((essay_scores, scores))
          essay_list = np.vstack((essay_list, wv_indices))
    if(is_test):
      essay_scores = None
    print max_words
    return essay_list, essay_scores

  def getData(self, dataset):
    if dataset == 0:
      if os.path.isfile(self.saved_data_essay_final):
        essays = np.load(self.saved_data_essay_final)
        scores = np.load(self.saved_data_score_final)
        arr = np.load(self.saved_data_score_nu_final)
        self.num_unique = arr[0]
      else:
        essays, scores = self.readInData(self.train_file, False)
        np.save(self.saved_data_essay, essays)
        np.save(self.saved_data_score, scores)
        temp_array = np.array([self.num_unique])
        np.save(self.saved_data_score_nu, temp_array)
    elif dataset == 1:
      essays, scores = self.readInData(self.test_file, True)
    else:
      essays, scores = self.readInData(self.valid_file, False)
    return essays, scores

  def getNumUnique(self):
    return self.num_unique