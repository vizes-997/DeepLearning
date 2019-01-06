import os
import sys
import time
from os import path

import numpy

import theano
import theano.tensor as T
import gzip
import cPickle

import matplotlib.pyplot as plt


def load_data(dataset):
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        data_dir = os.path.join(os.path.split(__file__)[0], "..", "data")
        if not path.exists(data_dir):
            print "No data directory to save data to. Try:"
            print "mkdir ../data"
            sys.exit(1)
        new_path = path.join(data_dir, data_file)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % url
        urllib.urlretrieve(url, dataset)

    
    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        train_set, valid_set, test_set = cPickle.load(f)
    print "size of the training set %s" % len(_make_array(train_set))
    print "size of the valid set %s" % len(_make_array(valid_set))
    print "size of the test set %s" % len(_make_array(test_set))
    return _make_array(train_set), _make_array(valid_set), _make_array(test_set)


def _make_array(xy):
    data_x, data_y = xy
    return zip(
        numpy.asarray(data_x, dtype=theano.config.floatX),
        numpy.asarray(data_y, dtype='int32'))


def _init_logreg_weights(n_hidden, n_out):
    weights = numpy.zeros((n_hidden, n_out), dtype=theano.config.floatX)
    bias = numpy.zeros((n_out,), dtype=theano.config.floatX)
    return (
        theano.shared(name='W', borrow=True, value=weights),
        theano.shared(name='b', borrow=True, value=bias)
    )


def _init_hidden_weights(n_in, n_out):
    rng = numpy.random.RandomState(1234)
    weights = numpy.asarray(
        rng.uniform(
            low=-numpy.sqrt(6. / (n_in + n_out)),
            high=numpy.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)
        ),
        dtype=theano.config.floatX
    )
    bias = numpy.zeros((n_out,), dtype=theano.config.floatX)
    return (
        theano.shared(value=weights, name='W', borrow=True),
        theano.shared(value=bias, name='b', borrow=True)
    )


def feed_forward(activation, weights, bias, input_):
    return activation(T.dot(input_, weights) + bias)

def sgd_step(param, cost, learning_rate):
    return param - (learning_rate * T.grad(cost, param))

# These are also symbolic.
def L1(L1_reg, w1, w2):
    return L1_reg * (abs(w1).sum() + abs(w2).sum())


def L2(L2_reg, w1, w2):
    return L2_reg * ((w1 ** 2).sum() + (w2 ** 2).sum())
 

def compile_model(n_in, n_classes, n_hidden, learning_rate, L1_reg, L2_reg):
    # allocate symbolic variables for the data
    x = T.vector('x')  # Features
    y = T.iscalar('y') # (Gold) Label
    
    # Allocate and initialize weights.  These are stored internally, and updated.
    hidden_W, hidden_b = _init_hidden_weights(n_in, n_hidden)
    logreg_W, logreg_b = _init_logreg_weights(n_hidden, n_classes)

    # Estimate P(y | x) given the current weights
    p_y_given_x = feed_forward(
                      T.nnet.softmax,
                      logreg_W,
                      logreg_b,
                      feed_forward(
                          T.tanh,
                          hidden_W,
                          hidden_b,
                          x)) # <--- Our input variable (the features)

    cost = (
        -T.log(p_y_given_x[0, y]) # <-- Negative log likelihood of gold label
        + L1(L1_reg, logreg_W, hidden_W)
        + L2(L2_reg, logreg_W, hidden_W)
    )

    # Compile the training function.  Successive calls to this update the weights.
    # Internal state is maintained.
    # The output is "cost", which requires the computation of p_y_given_x.  We
    # also define how to update the weights based on the input label.
    train_model = theano.function(
        inputs=[x, y],
        outputs=cost, 
        updates=[
            (logreg_W, sgd_step(logreg_W, cost, learning_rate)),
            (logreg_b, sgd_step(logreg_b, cost, learning_rate)),
            (hidden_W, sgd_step(hidden_W, cost, learning_rate)),
            (hidden_b, sgd_step(hidden_b, cost, learning_rate)),
        ]
    )

    # Compile the evaluation function, which returns a 0/1 loss wrt the true
    # label.
    evaluate_model = theano.function(
        inputs=[x, y],
        outputs=T.neq(y, T.argmax(p_y_given_x[0])),
    )
    return train_model, evaluate_model


def main(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=3,
             dataset='mnist.pkl.gz', n_hidden=500):
    train_examples, dev_examples, test_examples = load_data(dataset)
    print '-- COMPILING THE MODEL --'
    train_model, evaluate_model = compile_model(28*28, 10, n_hidden, learning_rate, L1_reg, L2_reg)
    print '-- TRAINING (%s iterations) --' % n_epochs
    epoch_list = []
    error_list = []
    for epoch in range(1, n_epochs+1):
        for x, y in train_examples:
            train_model(x, y)
        # compute zero-one loss on validation set
        error = numpy.mean([evaluate_model(x, y) for x, y in dev_examples])
        print('iteration %i, error perc %f %%' % (epoch, error * 100))
        epoch_list.append(epoch)
        error_list.append(error * 100)
    plt.plot(epoch_list, error_list)
    plt.ylabel('error percentage')
    plt.xlabel('iterations')
    plt.show()


if __name__ == '__main__':
    main()
