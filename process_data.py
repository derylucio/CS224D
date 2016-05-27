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
  def __init__(self, essay_info, train):
    self.id = essay_info[0]
    self.set = essay_info[1]
    self.content = essay_info[2]
    if (train):
	    if essay_info[5] == '':
	    	self.grade = (float)(essay_info[6])/2
	    else:
	    	self.grade = (float)(essay_info[6])/3
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
  	self.train = None
  	self.test = None
  	self.valid = None

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

  def readInData(self, filename, mode):
  	essay_list = []
  	with open(filename, 'rb') as tsv:
  		data = csv.reader(tsv, delimiter='\n')
  		iterdata = iter(data)
  		next(iterdata) # skip the first row
  		for row in iterdata:
  			row = row[0]
  			row.replace(" ", "")
  			entries = row.split("\t")
  			# entries = [x for x in entries if x != '']
  			# print entries, len(entries)
  			# print entries[6]
  			if mode == "train":
  				essay = Essay(entries, True)
  			else:
  				essay = Essay(entries, False)
  			# print "grade: ", essay.grade
  			essay_vector = np.zeros(self.N)
  			length = 0
  			for word in essay.content.split():
  				word = word.lower()
  				if word in self.W:
  					length += 1
  					essay_vector = essay_vector + self.W[word]
  			# print "length: ", length
  			essay.setVector(essay_vector/length)
  			essay_list.append(essay_vector)
  	if mode == "train":
  		self.train = essay_list
  	elif mode == "test":
  		self.test = essay_list
  	elif mode == "valid":
  		self.valid = essay_list
  	# print "n essays: ", len(essay_list)
	# return essay_list


if __name__ == '__main__':
  """
  The main function
  """
  N = 50
  processor = DataProcessor(N)
  processor.readTrainedWordVector(glove_file)
  # W, vocab, ivocab = processor.readTrainedWordVector(glove_file)
  processor.readInData(train_file, "train")
  processor.readInData(test_file, "test")
  processor.readInData(valid_file, "valid")
  # print processor.train
  # import cProfile
  # cProfile.run("runGames( **args )")
  pass