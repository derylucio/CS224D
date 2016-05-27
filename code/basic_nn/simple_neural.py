import os
import getpass
import sys
import time

import numpy as np
import tensorflow as tf
from q2_initialization import xavier_weight_init
import data_utils.utils as du
from utils import data_iterator
from model import LanguageModel

class Config(object):
  """Holds model hyperparams and data information.

  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation.
  """
  embed_size = 50
  batch_size = 64
  num_domains = 2
  hidden_size = 100
  max_epochs = 24
  early_stopping = 2
  dropout = 0.9
  lr = 0.001
  l2 = 0.0001
  window_size = 3

class EssayGraderModel(Model):

  def load_data(self, debug=False):
     """Loads starter word-vectors and train/dev/test data."""
    # Load the starter word vectors
    self.num_uniquewords #to recieve
    # Load the training set
    self.X_train, self.y_train #to receive
    if debug:
      self.X_train = self.X_train[:1024]
      self.y_train = self.y_train[:1024]

    # Load the dev set (for tuning hyperparameters)
    docs = du.load_dataset('data/ner/dev')
    self.X_dev, self.y_dev = #to receive
    if debug:
      self.X_dev = self.X_dev[:1024]
      self.y_dev = self.y_dev[:1024]

    # Load the test set (dummy labels only)
    self.X_test, self.y_test #to receive
   

  def add_placeholders(self):
    """Generate placeholder variables to represent the input tensors

    These placeholders are used as inputs by the rest of the model building
    code and will be fed data during training.  Note that when "None" is in a
    placeholder's shape, it's flexible

    Adds following nodes to the computational graph

    input_placeholder: Input placeholder tensor of shape
                       (None, window_size), type tf.int32
    labels_placeholder: Labels placeholder tensor of shape
                        (None, num_domains), type tf.float32
    dropout_placeholder: Dropout value placeholder (scalar),
                         type tf.float32

    Add these placeholders to self as the instance variables
  
      self.input_placeholder
      self.labels_placeholder
      self.dropout_placeholder

    (Don't change the variable names)
    """
    ### YOUR CODE HERE
    self.input_placeholder = tf.placeholder(tf.int32, shape=(None, None))
    self.labels_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.num_domains))
    self.dropout_placeholder = tf.placeholder(tf.float32)
    ### END YOUR CODE

  def create_feed_dict(self, input_batch, dropout, label_batch=None):
    """Creates the feed_dict for softmax classifier.

    A feed_dict takes the form of:

    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }


    Hint: The keys for the feed_dict should be a subset of the placeholder
          tensors created in add_placeholders.
    Hint: When label_batch is None, don't add a labels entry to the feed_dict.
    
    Args:
      input_batch: A batch of input data.
      label_batch: A batch of label data.
    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    ### YOUR CODE HERE
    if label_batch is None:
      label_batch = np.zeros((1, self.config.num_domains))
    feed_dict = {
      self.input_placeholder : input_batch,
      self.labels_placeholder : label_batch,
      self.dropout_placeholder : dropout
    }
    ### END YOUR CODE
    return feed_dict

  def add_embedding(self):
    #super simple model, just takes the mean of all the words as an indicator of the essay.
    with tf.device('/cpu:0'):
      ### YOUR CODE HERE
      L = tf.Variable(tf.convert_to_tensor(self.num_uniquewords, dtype=tf.float32))
      window = tf.nn.embedding_lookup(L, self.input_placeholder)
      window = tf.reduce_mean(window, 0)
      ### END YOUR CODE
      return window

  def add_model(self, window):
    """Adds the 1-hidden-layer NN.

    Hint: Use a variable_scope (e.g. "Layer") for the first hidden layer, and
          another variable_scope (e.g. "Softmax") for the linear transformation
          preceding the softmax. Make sure to use the xavier_weight_init you
          defined in the previous part to initialize weights.
    Hint: Make sure to add in regularization and dropout to this network.
          Regularization should be an addition to the cost function, while
          dropout should be added after both variable scopes.
    Hint: You might consider using a tensorflow Graph Collection (e.g
          "total_loss") to collect the regularization and loss terms (which you
          will add in add_loss_op below).
    Hint: Here are the dimensions of the various variables you will need to
          create

          W:  (window_size*embed_size, hidden_size)
          b1: (hidden_size,)
          U:  (hidden_size, num_domains)
          b2: (num_domains)

    https://www.tensorflow.org/versions/r0.7/api_docs/python/framework.html#graph-collections
    Args:
      window: tf.Tensor of shape (-1, window_size*embed_size)
    Returns:
      output: tf.Tensor of shape (batch_size, num_domains)
    """
    ### YOUR CODE HERE
    with tf.variable_scope('Layer') as hidden:
      W = tf.get_variable('W', (self.config.embed_size, self.config.hidden_size), initializer=xavier_weight_init())
      b1 = tf.get_variable('b1', (self.config.hidden_size,), initializer=xavier_weight_init())
      h = tf.tanh(tf.matmul(window, W)+ b1)
      with tf.variable_scope('Score') as score_scope:
        U = tf.get_variable('U', (self.config.hidden_size, self.config.num_domains), initializer=xavier_weight_init())
        h = tf.nn.dropout(h, self.dropout_placeholder)
        output = tf.matmul(h, U)
        regularization = self.config.l2*0.5*(tf.reduce_sum(tf.square(W)) + tf.reduce_sum(tf.square(U)))
        tf.add_to_collection('REGULARIZATION_LOSSES', regularization)


    ### END YOUR CODE
    return output 

  def add_loss_op(self, y):
    """Adds cross_entropy_loss ops to the computational graph.

    Hint: You can use tf.nn.softmax_cross_entropy_with_logits to simplify your
          implementation. You mght find tf.reduce_mean useful.
    Args:
      y: A tensor of shape (batch_size, n_classes)
    Returns:
      loss: A 0-d tensor (scalar)
    """
    ### YOUR CODE HERE
    reg = tf.get_collection("REGULARIZATION_LOSSES", scope='Layer/Softmax')[0]
    loss = tf.reduce_sum(tf.square_difference(y, self.labels_placeholder)) + reg
    ### END YOUR CODE
    return loss

  def add_training_op(self, loss):
    """Sets up the training Ops.

    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train. See 

    https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

    for more information.

    Hint: Use tf.train.AdamOptimizer for this model.
          Calling optimizer.minimize() will return a train_op object.

    Args:
      loss: Loss tensor, from cross_entropy_loss.
    Returns:
      train_op: The Op for training.
    """
    ### YOUR CODE HERE
    optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
    train_op = optimizer.minimize(loss)
    ### END YOUR CODE
    return train_op

  def __init__(self, config):
    """Constructs the network using the helper functions defined above."""
    self.config = config
    self.load_data(debug=False)
    self.add_placeholders()
    window = self.add_embedding()
    y = self.add_model(window)
    self.predictions = y
    self.loss = self.add_loss_op(y)
    self.train_op = self.add_training_op(self.loss)

  def run_epoch(self, session, input_data, input_labels,
                shuffle=True, verbose=True):
    orig_X, orig_y = input_data, input_labels
    dp = self.config.dropout
    # We're interested in keeping track of the loss and accuracy during training
    total_loss = []
    total_steps = len(orig_X) / self.config.batch_size
    for step, (x, y) in enG:umerate(
      data_iterator(orig_X, orig_y, batch_size=self.config.batch_size,
                   num_domains=self.config.num_domains, shuffle=shuffle)):
      feed = self.create_feed_dict(input_batch=x, dropout=dp, label_batch=y)
      loss, _ = session.run(
          [self.loss, self.train_op],
          feed_dict=feed)
      total_processed_examples += len(x)
      total_loss.append(loss)
      ##
      if verbose and step % verbose == 0:
        sys.stdout.write('\r{} / {} : loss = {}'.format(
            step, total_steps, np.mean(total_loss)))
        sys.stdout.flush()
    if verbose:
        sys.stdout.write('\r')
        sys.stdout.flush()
    return np.mean(total_loss)

  def predict(self, session, X, y=None):
    """Make predictions from the provided model."""
    # If y is given, the loss is also calculated
    # We deactivate dropout by setting it to 1
    dp = 1
    losses = []
    results = []
    if np.any(y):
        data = data_iterator(X, y, batch_size=self.config.batch_size,
                             num_domains=self.config.num_domains, shuffle=False)
    else:
        data = data_iterator(X, batch_size=self.config.batch_size,
                             num_domains=self.config.num_domains, shuffle=False)
    for step, (x, y) in enumerate(data):
      feed = self.create_feed_dict(input_batch=x, dropout=dp)
      if np.any(y):
        feed[self.labels_placeholder] = y
        loss, preds = session.run(
            [self.loss, self.predictions], feed_dict=feed)
        losses.append(loss)
      else:
        preds = session.run(self.predictions, feed_dict=feed)
      results.extend(preds)
    return np.mean(losses), results


def test_SimpleEssayGrader():
  """Test NER model implementation.

  You can use this function to test your implementation of the Named Entity
  Recognition network. When debugging, set max_epochs in the Config object to 1
  so you can rapidly iterate.
  """
  config = Config()
  with tf.Graph().as_default():
    model = EssayGraderModel(config)

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    with tf.Session() as session:
      best_val_loss = float('inf')
      best_val_epoch = 0

      session.run(init)
      for epoch in xrange(config.max_epochs):
        print 'Epoch {}'.format(epoch)
        start = time.time()
        ###
        train_loss = model.run_epoch(session, model.X_train,
                                                model.y_train)
        val_loss = model.predict(session, model.X_dev, model.y_dev)
        print 'Training loss: {}'.format(train_loss)
        print 'Training acc: {}'.format(train_acc)
        print 'Validation loss: {}'.format(val_loss)
        if val_loss < best_val_loss:
          best_val_loss = val_loss
          best_val_epoch = epoch
          if not os.path.exists("./weights"):
            os.makedirs("./weights")
        
          saver.save(session, './weights/ner.weights')
        if epoch - best_val_epoch > config.early_stopping:
          break
        ###
         print 'Total time: {}'.format(time.time() - start)
      
      saver.restore(session, './weights/ner.weights')
      print 'Test'
      print '=-=-='
      print 'Writing predictions to q2_test.predicted'
      res = model.predict(session, model.X_test, model.y_test)

if __name__ == "__main__":
  test_SimpleEssayGrader()
