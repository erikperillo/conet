#!/usr/bin/env python3

"""
Test your model with this script. Just call it passing an image as argument.
Example: ./predict.py path/to/maybe/cone.jpg
"""

import theano
import lasagne
from theano import tensor as T
import time
import numpy as np
import gzip
import pickle
from PIL import Image
import glob
import sys
import os


#model
MODEL_FILEPATH = "../data/conet_model.pkl"
#input images from dataset should have this shape (height, width)
INP_IMG_SHAPE = (42, 28)
#below values are respective to conet_dataset.pkl
X_MEAN = 136.23253
X_STD = 65.74257

def load_model(filepath, use_gzip=False):
    with (gzip.open if use_gzip else open)(filepath, "rb") as f:
        model = pickle.load(f)
    return model

def norm(img, u, sd):
    if img.dtype != np.float64:
        img = np.array(img, dtype=np.float64)
    return (img - u)/sd

def predict(network, img, pred_f):
    img = norm(img, X_MEAN, X_STD)
    img = img.reshape((1, 1) + img.shape)

    start_time = time.time()
    ret = pred_f(img)[0] == 1
    pred_time = time.time() - start_time

    return ret, pred_time

def main():
    """
    Reads image and tells wether it's a cone or not.
    """
    if len(sys.argv) < 2:
        print("usage: predict.py <img_or_imgsdir_path>")
        exit()

    print("loading model...")
    model = load_model(MODEL_FILEPATH)

    print("making prediction function...")
    inp = T.tensor4("inp")
    pred = lasagne.layers.get_output(model, inputs=inp, deterministic=True)
    pred = T.argmax(pred, axis=1)
    pred_f = theano.function([inp], pred)

    if os.path.isdir(sys.argv[1]):
        filepaths = glob.glob(os.path.join(sys.argv[1], "*"))
    else:
        filepaths = [sys.argv[1]]

    for fp in filepaths:
        print("in '{}'".format(fp))
        img = Image.open(fp).convert("L")
        img_shape = img.height, img.width
        if img_shape != INP_IMG_SHAPE:
            print("resizing image from {} to {}".format(img_shape,
                INP_IMG_SHAPE))
            img = img.resize(INP_IMG_SHAPE[::-1], Image.ANTIALIAS)
        img = np.asarray(img)

        pred, pred_time = predict(model, img, pred_f)
        print("is it a cone? %s" % ("YES!" if pred else "NO"))
        print("(processing time: %.6fs)" % pred_time)

if __name__ == "__main__":
    main()
