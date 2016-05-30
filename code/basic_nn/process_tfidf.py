import sys, csv, collections, os
import argparse, random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer
import string


train_file = "/Users/luciodery/Desktop/Stanford!/spring2015/CS224D/FinalProject/data/training_set.tsv"
test_file = "/Users/luciodery/Desktop/Stanford!/spring2015/CS224D/FinalProject/data/test_set.tsv"
valid_file = "/Users/luciodery/Desktop/Stanford!/spring2015/CS224D/FinalProject/data/valid_set.tsv"
prediction_file = "/Users/Chip/GitHub/CS224D/data/valid_sample_submission_5_column.csv"
glove_file = "/Users/Chip/GitHub/CS224D/models/glove.6B/glove.6B.50d.txt"
saved_tfid_essays = "/Users/luciodery/Desktop/Stanford!/spring2015/CS224D/FinalProject/code/basic_nn/saved_data/tfidf"
saved_tfid_essays_final = "/Users/luciodery/Desktop/Stanford!/spring2015/CS224D/FinalProject/code/basic_nn/saved_data/tfidf.npy"
saved_data_score = "/Users/luciodery/Desktop/Stanford!/spring2015/CS224D/FinalProject/code/basic_nn/saved_data/trainscores"
saved_data_score_final = "/Users/luciodery/Desktop/Stanford!/spring2015/CS224D/FinalProject/code/basic_nn/saved_data/trainscores.npy"

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
      #   self.grade = (float)(essay_info[6])/2
      # else:
      #   self.grade = (float)(essay_info[6])/3
      # if self.set == 1 or self.set == 2 or self.set == 5 or self.set == 6:
      #   self.grade = self.grade * 2
      # elif self.set == 3 or self.set == 4:
      #   self.grade = self.grade * 3
      # elif self.set == 8:
      #   self.grade = self.grade/3
  # if mode == 1:
  #   self.grade = 
    
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
class TFIDFProcessor:

  def __init__(self, N):
    self.N = N
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
    if(os.path.isfile(saved_tfid_essays_final)):
      self.train_tfidf_matrix = np.load(saved_tfid_essays_final)
      self.train_grade = np.load(saved_data_score_final)
    else:
      train, labels = self.readInData(train_file, 0)
      vectorizer = CountVectorizer(tokenizer=tokenize, stop_words='english', encoding="ISO-8859-1")
      train_matrix = vectorizer.fit_transform(train)
      tfidf = TfidfTransformer(norm="l2")
      tfidf.fit(train_matrix)
      self.train_tfidf_matrix = tfidf.transform(train_matrix)
      self.train_tfidf_matrix = self.train_tfidf_matrix.toarray()
      self.train_grade = np.array(labels)
      np.save(saved_tfid_essays, self.train_tfidf_matrix)
      np.save(saved_data_score, self.train_grade)
    # self.readTrainedWordVector(glove_file)
    # self.convertToTrain()

  def readPrediction(self, filename):
    prediction = {}
    with open(filename, 'rb') as csvin:
      data = csv.reader(csvin, delimiter='\n')
      iterdata = iter(data)
      next(iterdata) # skip the first row
      for row in iterdata:
        row = row[0].split(',')

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

  # train = essays
  def readInData(self, filename, mode):
    essays = []
    grades = []
    with open(filename, 'rb') as tsv:
      data = csv.reader(tsv, delimiter='\n')
      iterdata = iter(data)
      next(iterdata) # skip the first row
      for row in iterdata:
        row = row[0]
        row.replace(" ", "")
        entries = row.split("\t")
        if mode == 0:
          essay = Essay(entries, True)
        else:
          essay = Essay(entries, False)
        if essay.set > 6:
          break
        essays.append(essay.content)
        grades.append(essay.grade)
      tsv.close()
    return essays, grades

stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
  stemmed = []
  for item in tokens:
    stemmed.append(stemmer.stem(item))
  return stemmed

def tokenize(text):
  text = "".join([ch for ch in text if ch not in string.punctuation])
  # print "text: ", text
  tokens = nltk.word_tokenize(text)
  stems = stem_tokens(tokens, stemmer)
  return stems

if __name__ == '__main__':
  """
  The main function
  """
  N = 50
  processor = DataProcessor(N)
  train, labels = processor.readInData(train_file, 0)
  stemmer = PorterStemmer()
  vectorizer = CountVectorizer(tokenizer=tokenize, stop_words='english', encoding="ISO-8859-1")
  train_matrix = vectorizer.fit_transform(train)
  vocab = vectorizer.vocabulary_
  print "vocab size: ", len(vocab)
  tfidf = TfidfTransformer(norm="l2")
  tfidf.fit(train_matrix)
  tf_idf_matrix = tfidf.transform(train_matrix) # the train tfidf matrix
  grade = np.array(labels) # the label
  # print tf_idf_matrix.shape
  # print tf_idf_matrix
  pass