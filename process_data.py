import sys, csv, collections

train_file = "data/train_small.tsv"
test_file = "data/test_small.tsv"
valid_file = "data/valid_small.tsv"

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

def readInData(filename):
	with open(filename, 'rb') as tsv:
		data = csv.reader(tsv, delimiter='\n')
		iterdata = iter(data)
		for row in iterdata:
			print "row: ", row

	# print data


if __name__ == '__main__':
  """
  The main function
  """
  train_essays = readInData(train_file)

  # import cProfile
  # cProfile.run("runGames( **args )")
  pass