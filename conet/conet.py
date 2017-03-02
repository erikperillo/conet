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
#cnn implements all the network stuff.
#sgd implements stochastic gradient descent, which is used to train the net.
#both of them are built on top of theano.
import cnn
import sgd
import theano
#other stuff we need
from theano import tensor
import numpy as np
import gzip
import pickle


#input dataset path
DATASET_FILEPATH = "../data/cones_dataset.pkl"
#output model
OUTPUT_MODEL_FILEPATH = "../data/conet_model.pkl"
#train fraction
TR_FRAC = 0.8
#cross-validation fraction
CV_FRAC = 0.18
#input images from dataset should have this shape (height, width)
inp_img_shape = (42, 28)


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

def train_net():
    """
    Trains network on images dataset.
    """

    #data loading
    print("loading data...")
    X, y = load_dataset(DATASET_FILEPATH)
    print("\tX, y shapes:", X.shape, y.shape)

    #train, cv, test splits
    print("splitting train, validation, test sets...")
    X_tr, y_tr, X_cv, y_cv, X_te, y_te = tr_cv_te_split(X, y, TR_FRAC, CV_FRAC)
    print("\ttrain shape:", X_tr.shape, y_tr.shape)
    print("\tcv shape:", X_cv.shape, y_cv.shape)
    print("\ttest shape:", X_te.shape, y_te.shape)

    #symbolic variables
    x = tensor.matrix(name="x")
    y = tensor.ivector(name="y")

    #convolutional neural network parameters.
    #first layer. extracts basic low-level features
    layer_0_params = (
        #convolution
        {
            #assumes grayscale input images
            "n_inp_maps": 1,
            #input maps shape
            "inp_maps_shape": inp_img_shape,
            #number of feature maps that get out of this layer
            "n_out_maps": 16,
            "filter_shape": (3, 3),
        },
        #max-pooling
        {
            "shape": (2, 2)
        }
    )
    #second layer. extracts higher-level features
    layer_1_params = (
        #convolution
        {
            "n_out_maps": 32,
            "filter_shape": (5, 5),
        },
        #max-pooling
        {
            "shape": (2, 2)
        }
    )
    #the final layer is a fully-connected neural network.
    fully_connected_layer_params = {
        #number of hidden units
        "n_hidden": 196,
        #1 class -> two outputs (true or false)
        "n_out": 2
    }

    #building neural network
    inp = x.reshape((x.shape[0], 1,) + inp_img_shape)
    clf = cnn.ConvolutionalNeuralNetwork(
        inp=inp,
        conv_pool_layers_params=[
            layer_0_params,
            layer_1_params,],
        fully_connected_layer_params=fully_connected_layer_params)

    #building theano shared variables, this binds real data to symbolic vars
    X_tr_sh = theano.shared(X_tr, borrow=True)
    y_tr_sh = theano.shared(y_tr, borrow=True)
    X_cv_sh = theano.shared(X_cv, borrow=True)
    y_cv_sh = theano.shared(y_cv, borrow=True)

    #stochastic gradient descent with validation check.
    print("calling sgd_with_validation", flush=True)
    sgd.sgd_with_validation(clf,
        X_tr_sh, y_tr_sh, X_cv_sh, y_cv_sh,
        learning_rate=0.001, reg_term=0.002,
        batch_size=10, n_epochs=12,
        max_its=20000, improv_thresh=0.01, max_its_incr=4,
        x=x,
        rel_val_tol=None,
        val_freq="auto",
        verbose=True)

    #building accuracy function
    #acc = theano.function([inp, y], clf.score(y))
    #acc_val = acc(np.reshape(X_te, (X_te.shape[0], 1,) + inp_img_shape), y_te)
    #print("accuracy: %.2f%%" % (100*acc_val))

    #saving model
    if OUTPUT_MODEL_FILEPATH:
        print("saving model...")
        with open(OUTPUT_MODEL_FILEPATH, "wb") as f:
            pickle.dump(clf, f)

if __name__ == "__main__":
    train_net()
