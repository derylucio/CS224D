import sys, csv, collections
import argparse
import numpy as np

train_file = "data/train_small.tsv"
test_file = "data/test_small.tsv"
valid_file = "data/valid_small.tsv"
glove_file = "models/glove.6B/glove.6B.50d.txt"
ESSAY_INDEX = 2
ESSAY_ID_INDEX = 0
DOMAIN_1_INDEX = 6

class Essay:
  """
  Each represents information associated with an essay
  28 fields. Refer to Kaggle's data page
  """
  def __init__(self, essay_info):
    self.id = essay_info[0]
    self.set = essay_info[1]
    self.content = essay_info[2]
    # self.rate1 = 
    self.vector = None

  def setVector(self, v):
  	self.vector = v

# train_essays = dict()
# test_essays = dict()
# valid_essays = dict()
W = {}
vocab = dict()
ivocab = dict()

"""
	W is a dict word:word_vector (word_vector is a list)
	vocab is a dict word:index
	ivocab index:word
"""

#will need to unpack the information.
class DataProcessor:

  def __init__(self, N):
  	self.N = N
  	self.W = None
  	self.vocab = None
  	self.ivocab = None

  def readTrainedWordVector(self, filename):
  	parser = argparse.ArgumentParser()
  	parser.add_argument('--vectors_file', default=filename, type=str)
  	args = parser.parse_args()
  	index = 0
  	words = {}
	vocab = {}
	ivocab = {}
  	with open(args.vectors_file, 'r') as f:
  		for line in f:
  			vals = line.rstrip().split(' ')
  			word = vals[0].lower()
  			index = index + 1
  			vocab[word] = index
  			ivocab[index] = word
  			words[word] = np.array([float(x) for x in vals[1:]])
	# return (W, vocab, ivocab)
	self.W = words
	self.vocab = vocab
	self.ivocab = ivocab

  def readInData(self, filename):
  	essay_list = []
  	with open(filename, 'rb') as tsv:
  		data = csv.reader(tsv, delimiter='\n')
  		iterdata = iter(data)
  		next(iterdata) # skip the first row
  		for row in iterdata:
  			row = row[0]
  			row.replace(" ", "")
  			entries = row.split("\t")
  			# print entries, len(entries)
  			essay = Essay(entries)
  			essay_vector = np.zeros(self.N)
  			length = 0
  			# print essay.content.split()
	        for word in essay.content.split():
	        	word = word.lower()
	        	print word
	        	if word in self.W:
	        		print "in W"
	        		length += 1
	        		essay_vector = essay_vector + self.W[word]
	        # print "length: ", length
	        essay.setVector(essay_vector/length)
	        essay_list.append(essay_vector)
	return essay_list


if __name__ == '__main__':
  """
  The main function
  """
  N = 50
  processor = DataProcessor(N)
  processor.readTrainedWordVector(glove_file)
  # W, vocab, ivocab = processor.readTrainedWordVector(glove_file)
  print "done"
  train_essays = processor.readInData(train_file)
  test_essays = processor.readInData(test_file)
  valid_essays = processor.readInData(valid_file)
  # import cProfile
  # cProfile.run("runGames( **args )")
  pass