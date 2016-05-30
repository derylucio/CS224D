import sys, csv, collections, os
import argparse, random
import numpy as np

train_file = "/Users/Chip/GitHub/CS224D/data/training_set_rel3.tsv"
test_file = "/Users/Chip/GitHub/CS224D/data/test_small.tsv"
valid_file = "/Users/Chip/GitHub/CS224D/data/valid_small.tsv"
prediction_file = "/Users/Chip/GitHub/CS224D/data/valid_sample_submission_5_column.csv"
glove_file = "/Users/Chip/GitHub/CS224D/models/glove.6B/glove.6B.50d.txt"
ESSAY_INDEX = 2
ESSAY_ID_INDEX = 0
DOMAIN_1_INDEX = 6
TRAIN = 0
VALID = 1
TEST = 2

"""
resolved score range:
1: 2 - 12
2: 1 - 4
3: 0 - 3
4: 0 - 3
5: 0 - 4
6: 0 - 4
7: 0 - 30
8: 0 - 60

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
    self.grade = (float)(essay_info[6])
    if mode == 0:
    	self.grade = (float)(essay_info[6])
	    # if essay_info[5] == '':
	    # 	self.grade = (float)(essay_info[6])/2
	    # else:
	    # 	self.grade = (float)(essay_info[6])/3
	    # if self.set == 1 or self.set == 2 or self.set == 5 or self.set == 6:
	    # 	self.grade = self.grade * 2
	    # elif self.set == 3 or self.set == 4:
	    # 	self.grade = self.grade * 3
	    # elif self.set == 8:
	    # 	self.grade = self.grade/3
	# if mode == 1:
	# 	self.grade = 
    
    self.vector = None

  def setVector(self, v):
  	self.vector = v

  def getVector(self):
  	return self.vector

  def getGrade(self):
  	return self.grade

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
  	self.pred = None
  	self.vocab_size = None
  	self.nTrain = None
  	self.nValid = None
  	self.nTest = None
  	self.trainX = None
  	self.trainY = None
  	self.validX = None
  	self.validY = None
  	self.testX = None
  	self.testY = None
    # if dataset == 0:
    #   if os.path.isfile(self.saved_data_essay_final):
    #     essays = np.load(self.saved_data_essay_final)
    #     scores = np.load(self.saved_data_score_final)
    #     arr = np.load(self.saved_data_score_nu_final)
    #     self.num_unique = arr[0]
    #   else:
    #     essays, scores = self.readInData(self.train_file, False)
    #     np.save(self.saved_data_essay, essays)
    #     np.save(self.saved_data_score, scores)
    #     temp_array = np.array([self.num_unique])
    #     np.save(self.saved_data_score_nu, temp_array)
  	self.readTrainedWordVector(glove_file)
  	self.readInData(train_file, 0)
  	self.convertToTrain()

  def readPrediction(self, filename):
  	prediction = {}
  	with open(filename, 'rb') as csvin:
  		data = csv.reader(csvin, delimiter='\n')
  		iterdata = iter(data)
  		next(iterdata) # skip the first row
  		for row in iterdata:
  			row = row[0].split(',')
    csvin.close()	

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

  	self.vocab_size = len(words)
  	# vector_dim = len(words[ivocab[0]]) # the same as N
  	W = np.zeros((self.vocab_size, self.N))
  	for word, v in words.items():
  		W[vocab[word], :] = v
  	W_norm = np.zeros(W.shape)
  	d = (np.sum(W ** 2, 1) ** (0.5))
  	W_norm = (W.T / d).T

  	self.W = W_norm
  	self.vocab = vocab
  	self.ivocab = ivocab

  def convertToTrain(self):
    train_X = np.zeros((self.nTrain, self.N))
    train_Y = np.zeros((self.nTrain, 1))
    index = 0
    for e_id, e in self.train.items():
      train_X[index, :] = e.getVector()
      train_Y[index] = e.getGrade()
      index += 1
      # print "a: ", train_Y[index]
    self.trainX = train_X
    self.trainY = train_Y
    # print train_X
    # print train_Y
    valid_X = np.zeros((self.nValid, self.N))
    valid_Y = np.zeros((self.nValid, 1))
    index = 0
    for e_id, e in self.valid.items():
      valid_X[index, :] = e.getVector()
      valid_Y[index] = e.getGrade()
      index += 1
    self.validX = valid_X
    self.validY = valid_Y
    test_X = np.zeros((self.nTest, self.N))
    test_Y = np.zeros((self.nTest, 1))
    index = 0
    for e_id, e in self.test.items():
      test_X[index, :] = e.getVector()
      test_Y[index] = e.getGrade()
      index += 1
    self.testX = test_X
    self.testY = test_Y


  def readInData(self, filename, mode):
    essays = []
  	# train_list = {}
  	# valid_list = {}
  	# test_list = {}
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
  			if mode == 0:
  				essay = Essay(entries, True)
  			else:
  				essay = Essay(entries, False)
  			# print "essay set: ", essay.set
  			# print "grade: ", essay.grade
  			if essay.set > 6:
  				# print "end of 6"
  				break
  			essay_vector = np.zeros(self.N)
  			length = 0
  			for word in essay.content.split():
  				word = word.lower()
  				if word in self.vocab:
  					length += 1
  					# print self.W[self.vocab[word]]
  					essay_vector = essay_vector + self.W[self.vocab[word]]
  					# essay_vector = essay_vector + self.W[word]
  			# if length > 550:
  			# 	print entries
  			essay.setVector(essay_vector/length)
  			if bool(random.getrandbits(1)):
  				train_list[essay.id] = essay
  			else:
  				if bool(random.getrandbits(1)):
  					if bool(random.getrandbits(1)):
  						train_list[essay.id] = essay
  					else:
  						valid_list[essay.id] = essay
  				else:
  					if bool(random.getrandbits(1)):
  						train_list[essay.id] = essay
  					else:
  						test_list[essay.id] = essay
  	
  	self.train = train_list
  	self.valid = valid_list
  	self.test = test_list
  	self.nTrain = len(self.train)
  	self.nValid = len(self.valid)
  	self.nTest = len(self.test)
  	print "# train sample: ", self.nTrain
  	print "# valid sample: ", self.nValid
  	print "# test sample: ", self.nTest
  			# essay_list.append(essay_vector)
  	# if mode == 0:
  	# 	self.train = essay_list
  	# elif mode == 1:
  	# 	self.test = essay_list
  	# elif mode == "valid":
  	# 	self.valid = essay_list
  	# print "n essays: ", len(essay_list)
	# return essay_list

if __name__ == '__main__':
  """
  The main function
  """
  N = 50
  processor = DataProcessor(N)
  # processor.readTrainedWordVector(glove_file)
  # print processor.vocab
  # print processor.W[9]
  # W, vocab, ivocab = processor.readTrainedWordVector(glove_file)
  # processor.readInData(train_file, 0)
  # processor.convertToTrain()
  # print "# train sample: ", processor.trainX.shape
  # print "# valid sample: ", processor.validX.shape
  # print "# test sample: ", processor.testX.shape
  # print "train Y: ", processor.trainY
  # processor.readPrediction(prediction_file)
  # processor.readInData(valid_file, 1)
  # processor.readInData(valid_file, "valid")
  # print processor.train
  # import cProfile
  # cProfile.run("runGames( **args )")
  pass