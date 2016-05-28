import sys, csv, collections
import argparse
import numpy as np

train_file = "data/training_set_rel3.tsv"
test_file = "data/test_small.tsv"
valid_file = "data/valid_small.tsv"
glove_file = "models/glove.6B/glove.6B.50d.txt"
ESSAY_INDEX = 2
ESSAY_ID_INDEX = 0
DOMAIN_1_INDEX = 6
TRAIN = 0
VALID = 1
TEST = 2

"""
1: 0 - 6
2: 0 - 6
3: 0 - 4
4: 0 - 4
5: 0 - 6
6: 0 - 6
7: 0 - 12
8: 0 - 36

normalize all grades to scale 0 - 12

"""
class Essay:
  """
  Each represents information associated with an essay
  28 fields. Refer to Kaggle's data page
  """
  def __init__(self, essay_info, mode):
    self.id = (int)(essay_info[0])
    self.set = (int)(essay_info[1])
    self.content = essay_info[2]
    if mode == 0:
	    if essay_info[5] == '':
	    	self.grade = (float)(essay_info[6])/2
	    else:
	    	self.grade = (float)(essay_info[6])/3
	    if self.set == 1 or self.set == 2 or self.set == 5 or self.set == 6:
	    	self.grade = self.grade * 2
	    elif self.set == 3 or self.set == 4:
	    	self.grade = self.grade * 3
	    elif self.set == 8:
	    	self.grade = self.grade/3
    
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
  			if word == '<unk>':
  				continue
  			vocab[word] = index
  			# if index == 0:
  			# 	print vocab
  			ivocab[index] = word
  			index = index + 1
  			words[word] = np.array([float(x) for x in vals[1:]])
  			# if index == 10:
  			# 	print words[word]
	# normalize correctly now

	vocab_size = len(words)
	vector_dim = len(words[ivocab[0]])
	W = np.zeros((vocab_size, vector_dim))
	for word, v in words.items():
		W[vocab[word], :] = v
	W_norm = np.zeros(W.shape)
	d = (np.sum(W ** 2, 1) ** (0.5))
	W_norm = (W.T / d).T

	self.W = W_norm
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
  			# print "essay set: ", essay.set
  			# print "grade: ", essay.grade
  			# if essay.set > 6:
  			# 	print "end of 6"
  			# 	break
  			essay_vector = np.zeros(self.N)
  			length = 0
  			for word in essay.content.split():
  				word = word.lower()
  				if word in self.vocab:
  					length += 1
  					print self.W[self.vocab[word]]
  					essay_vector = essay_vector + self.W[self.vocab[word]]
  					# essay_vector = essay_vector + self.W[word]
  			if length > 550:
  				print entries
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
  # print processor.vocab
  # print processor.W[9]
  # W, vocab, ivocab = processor.readTrainedWordVector(glove_file)
  processor.readInData(train_file, "train")
  # processor.readInData(test_file, "test")
  # processor.readInData(valid_file, "valid")
  # print processor.train
  # import cProfile
  # cProfile.run("runGames( **args )")
  pass