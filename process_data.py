import sys, csv, collections
import numpy as np

train_file = "data/train_small.tsv"
test_file = "data/test_small.tsv"
valid_file = "data/valid_small.tsv"
ESSAY_INDEX = 2
ESSAY_ID_INDEX = 0
DOMAIN_1_INDEX = 6

class Essay:
  """
  Each represents information associated with an essay
  """
  def __init__(self, essay_id, grades):
    self.grades = grades

  def test():
  	print "in test"

train_essays = dict()
test_essays = dict()
valid_essays = dict()

#will need to unpack the information.
class DataProcessor:

  def __init__(self):
    self.word_index_map = {}
    self.num_unique = 0

  def readInData(self, filename):
    essay_list = []
  	with open(filename, 'rb') as tsv:
  		data = csv.reader(tsv, delimiter='\n')
  		iterdata = iter(data)
  		for row in iterdata:
        wv_indices = []
  			entries = row.split("\t")
        essay = entries[ESSAY_INDEX]
        words = essay.split()
        for word in words:
          if word in self.word_index_map:
            wv_indices.append(self.word_index_map[word])
          else:
            self.word_index_map[word] = self.num_unique
            self.num_unique += 1
        essay_list.append(np.array(wv_indices))
    return essay_list, essay_scores
	# print data


if __name__ == '__main__':
  """
  The main function
  """
  processor = DataProcessor()
  train_essays, train_scores = processor.readInData(train_file)
  test_essays, test_scores = processor.readInData(test_file)
  valid_essays, valid_scores = processor.readInData(valid_file)
  # import cProfile
  # cProfile.run("runGames( **args )")
  pass