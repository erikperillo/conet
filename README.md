# CONET - Convolutional Neural Network for cones

## Requirements
- theano
- numpy
- lasagne
- PIL/pillow (optional)

## Files
- conet/conet.py: Assembling the model and training.
- conet/trloop.py: Training loop routine with validation.
- conet/predict.py: Play with your trained model.
- data/cones_dataset.pkl: Augmented dataset of ~3800 images of cones/not cones.

## Using
You must first *train* your model and then use it to make *predictions*.

### Training
In the training step, you define the architecture of your network and then
make it learn what is a cone by giving it thousands of samples.

**How to do it:**
- make sure you have the dataset in *data/cones_dataset.pkl*.
- enter the *conet* dir and simply run *./conet.py*.

After a while you'll enter the training loop, where training error will be
reported for each batch of images. Hopefully the validation accuracy will
increase after each iteration until it's good enough.

### Using
Once the training process is done, your model is saved as something like
*./data/conet_model.pkl*. Enter conet dir and simply run
*./predict.py <img_or_dir_path>*, where *<img_or_dir_path>* is either the path
to a image or the path of a directory containing only images.
It'll iterate over all images telling whether it's a cone or not.
