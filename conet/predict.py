#!/usr/bin/env python3

"""
Test your model with this script. Just call it passing an image as argument.
Example: ./predict.py path/to/maybe/cone.jpg
"""

import cnn
import theano
from theano import tensor
import numpy as np
import gzip
import pickle
import cv2
import sys


#model
MODEL_FILEPATH = "../data/conet_model.pkl"
#input images from dataset should have this shape (height, width)
INP_IMG_SHAPE = (42, 28)

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

def predict(model, img):
    pred = theano.function([model.inp], model.pred)
    img = norm(img, X_MEAN, X_STD)
    x = img.reshape((1, 1) + img.shape)
    return pred(x)[0] == 1

def main():
    """
    Reads image and tells wether its a cone or not.
    """
    if len(sys.argv) < 2:
        print("usage: predict.py <img_filepath>")
        exit()

    print("loading model...")
    model = load_model(MODEL_FILEPATH)

    print("loading image...")
    filepath = sys.argv[1]
    img = cv2.imread(filepath, 0)
    if img.shape != INP_IMG_SHAPE:
        print("resizing image from {} to {}".format(img.shape, INP_IMG_SHAPE))
        img = cv2.resize(img, INP_IMG_SHAPE[::-1])

    pred = predict(model, img)
    print("is it a cone? %s" % ("YES!" if pred else "NO"))

if __name__ == "__main__":
    main()