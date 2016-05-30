from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, TimeDistributed
from keras.layers import Input, Embedding, LSTM
from keras.layers import Merge
import numpy as np
import keras.callbacks
from keras.regularizers import l2
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
import sys
sys.path.insert(0, '/Users/luciodery/Desktop/Stanford!/spring2015/CS224D/FinalProject/code/basic_nn')
from process_data import DataProcessor
from process_tfidf import TFIDFProcessor
import time

TRAIN_FRACT = 0.9
EMBEDDING_DIM = 50
MAX_ESSAY_LENGTH = 950
HIDDEN_DIM = 50
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 20
VALIDATION_SPLIT = 0.3
EARLY_STOPPING = 2
PCA_DIMENSION = 500
OUTPUT_DIM = 32


#Get the data
data_processor = DataProcessor()
X_train, Y_train = data_processor.getData(0)
num_uniquewords = data_processor.getNumUnique() 
perm = np.random.permutation(X_train.shape[0]);

#get the tfidf data
tfidf_processor = TFIDFProcessor(50)
np_tfidf = tfidf_processor.train_tfidf_matrix

X_train = X_train[perm, :]
Y_train = Y_train[perm] 
num_train = int(TRAIN_FRACT*X_train.shape[0])

x_train = X_train[:num_train]
y_train = Y_train[:num_train]
x_test = X_train[num_train:]
y_test = Y_train[num_train:]

np_tfidf = np_tfidf[perm,:]
np_tfidf_train = np_tfidf[:num_train]
np_tfidf_test = np_tfidf[num_train:]
# #Building the deep bidirectional neural netowrk

# #Replicate the model
# def fork(model, n=2):
# 	forks = []
# 	for i in range(n):
# 		f = Sequential()
# 		f.add(model)
# 		forks.append(f)
# 	return forks


# forward1 = Sequential() 
# forward1.add(Embedding(num_uniquewords + 1, EMBEDDING_DIM, input_length=MAX_ESSAY_LENGTH, mask_zero=True))
# forward1.add(LSTM(output_dim=HIDDEN_DIM, init='uniform', inner_init='uniform',
#                forget_bias_init='one', return_sequences=True, activation='tanh',
#                inner_activation='sigmoid'))


# backward1 = Sequential()
# backward1.add(Embedding(num_uniquewords + 1, EMBEDDING_DIM, input_length=MAX_ESSAY_LENGTH, mask_zero=True))
# backward1.add(LSTM(output_dim=HIDDEN_DIM, init='uniform', inner_init='uniform',
#                forget_bias_init='one', return_sequences=True, activation='tanh',
#                inner_activation='sigmoid', go_backwards=True))

# merged1 = Sequential()
# merged1.add(Merge([forward1, backward1], mode='sum'))

# forward2, backward2 = fork(merged1, 2)
# forward2.add(LSTM(output_dim=HIDDEN_DIM, init='uniform', inner_init='uniform',
#                forget_bias_init='one', return_sequences=True, activation='tanh',
#                inner_activation='sigmoid'))
# backward2.add(LSTM(output_dim=HIDDEN_DIM, init='uniform', inner_init='uniform',
#                forget_bias_init='one', return_sequences=True, activation='tanh',
#                inner_activation='sigmoid', go_backwards=True))

# merged2 = Merge([forward1, backward1], mode='sum')
# final_lstm = Sequential()
# final_lstm.add(merged2)
# final_lstm.add(LSTM(output_dim=HIDDEN_DIM, init='uniform', inner_init='uniform',
#                forget_bias_init='one', activation='tanh',
#                inner_activation='sigmoid'))

# #Add TFIDF Features
# #TFIDF to campute important document info
# tfidf = Sequential()
# tfidf.add(Dense(PCA_DIMENSION, input_dim=np_tfidf.shape[1]))
# tfidf.add(Dense(HIDDEN_DIM))

# #Merge and Predict
# merged = Merge([final_lstm, tfidf], mode='concat')
# final_model = Sequential()
# final_model.add(merged)
# final_model.add(Dense(1))

# optim = RMSprop(lr=LEARNING_RATE)
# final_model.compile(loss='mean_squared_error', optimizer=optim, metrics=['mse'])
# early_stopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING, verbose=0, mode='min')
# final_model.fit([x_train, x_train, np_tfidf_train], y_train, batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS, validation_split=VALIDATION_SPLIT, callbacks=[early_stopper])
# score =  final_model.evaluate([x_test, x_test, np_tfidf_test], y_test, batch_size=BATCH_SIZE)
# print score


forward1 = Sequential() 
forward1.add(Embedding(num_uniquewords + 1, EMBEDDING_DIM, input_length=MAX_ESSAY_LENGTH, mask_zero=True))
forward1.add(LSTM(output_dim=HIDDEN_DIM, init='uniform', inner_init='uniform',
               forget_bias_init='one', activation='tanh',
               inner_activation='sigmoid'))


backward1 = Sequential()
backward1.add(Embedding(num_uniquewords + 1, EMBEDDING_DIM, input_length=MAX_ESSAY_LENGTH, mask_zero=True))
backward1.add(LSTM(output_dim=HIDDEN_DIM, init='uniform', inner_init='uniform',
               forget_bias_init='one', activation='tanh',
               inner_activation='sigmoid', go_backwards=True))

merged1 = Sequential()
merged1.add(Merge([forward1, backward1], mode='sum'))

# #Add TFIDF Features
# #TFIDF to campute important document info
# tfidf = Sequential()
# tfidf.add(Dense(PCA_DIMENSION, input_dim=np_tfidf.shape[1]))
# tfidf.add(Dense(HIDDEN_DIM))

# #Merge and Predict
# merged = Merge([merged1, tfidf], mode='concat')
final_model = Sequential()
final_model.add(merged1)
final_model.add(Dense(1))


optim = RMSprop(lr=LEARNING_RATE)
final_model.compile(loss='mean_squared_error', optimizer=optim, metrics=['mse'])
early_stopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING, verbose=0, mode='min')
final_model.fit([x_train, x_train], y_train, batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS, validation_split=VALIDATION_SPLIT, callbacks=[early_stopper])
score =  final_model.evaluate([x_test, x_test], y_test, batch_size=BATCH_SIZE)
print score
