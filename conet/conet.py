#!/usr/bin/env python3

"""
CONET - A Convolutional Neural Network for Cones.

This scripts takes the dataset with cone images,
assembles the network and trains it with the data.
The model is saved so you can use it later.

Some amazing sources to learn more about the wonders of CNNs:
    - https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
    - http://deeplearning.net/tutorial/lenet.html
    - http://cs231n.github.io/convolutional-networks/
"""

#these are the three main modules.
#trloop implements gradient descent training loop.
#theano is the basis for everything.
#lasagne sits on top of theano and complements it.
import trloop
import theano
import lasagne
#other stuff we need
from theano import tensor as T
import numpy as np
import gzip
import pickle


#input dataset path
DATASET_FILEPATH = "../data/cones_dataset.pkl"
#output model
OUTPUT_MODEL_FILEPATH = "../data/conet_model.pkl"
#train fraction
TR_FRAC = 0.9
#cross-validation fraction
CV_FRAC = 0.075
#input images from dataset should have this shape (height, width)
INPUT_SHAPE = (42, 28)


def load_dataset(filepath, use_gzip=False):
    with (gzip.open if use_gzip else open)(filepath, "rb") as f:
        X, y = pickle.load(f)

    return np.array(X, dtype="float64"), np.array(y, dtype="int32")

def tr_cv_te_split(X, y, tr_frac, cv_frac):
    assert X.shape[0] == y.shape[0]

    n = y.shape[0]
    tr = int(n*tr_frac)
    cv = int(n*cv_frac)

    return X[:tr], y[:tr], X[tr:tr+cv], y[tr:tr+cv], X[tr+cv:], y[tr+cv:]

def build_cnn(input_shape, input_var=None):
    network = lasagne.layers.InputLayer(shape=input_shape,
                                        input_var=input_var)

    #input shape in form n_batches, depth, rows, cols
    network = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)

    # Convolutional layer with 20 kernels of size 3x3. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=8, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=16, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of x units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=96,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


def train_net():
    """
    Trains network on images dataset.
    """

    #data loading
    print("loading data...")
    X, y = load_dataset(DATASET_FILEPATH)
    X = X.reshape((X.shape[0], 1) + INPUT_SHAPE)
    print("\tX, y shapes:", X.shape, y.shape)

    #train, cv, test splits
    print("splitting train, validation, test sets...")
    X_tr, y_tr, X_cv, y_cv, X_te, y_te = tr_cv_te_split(X, y, TR_FRAC, CV_FRAC)
    print("\ttrain shape:", X_tr.shape, y_tr.shape)
    print("\tcv shape:", X_cv.shape, y_cv.shape)
    print("\ttest shape:", X_te.shape, y_te.shape)

    #symbolic variables
    input_var = T.tensor4(name="inputs")
    target_var = T.ivector(name="targets")

    print("building network...")
    #input_var = input_var.reshape((input_var.shape[0], 1) + INPUT_SHAPE)
    network = build_cnn((None, 1) + INPUT_SHAPE, input_var)

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    #reg = lasagne.regularization.regularize_network_params(network,
    #    lasagne.regularization.l2)
    #loss += reg*0.0001

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
        target_var)
    test_loss = test_loss.mean()
    #test_loss = test_loss + reg*0.001
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
        dtype=theano.config.floatX)

    print("compiling functions...", flush=True)
    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    print("calling loop")
    try:
        trloop.train_loop(
            X_tr, y_tr, train_fn,
            n_epochs=10, batch_size=10,
            X_val=X_cv, y_val=y_cv, val_f=val_fn,
            val_acc_tol=None,
            max_its=None,
            verbose=2)
    except KeyboardInterrupt:
        print("Ctrl+C pressed.")
        pass

    print("end.")
    err, acc = val_fn(X_te, y_te)
    print("test loss: %f | test acc: %f" % (err, acc))

    if OUTPUT_MODEL_FILEPATH:
        print("saving model...")
        with open(OUTPUT_MODEL_FILEPATH, "wb") as f:
            pickle.dump(network, f)

if __name__ == "__main__":
    train_net()
