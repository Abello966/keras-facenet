import re
import tensorflow as tf
import keras as kr
import pickle as pkl
import inception_resnet_v1 as inception

weights = pkl.load(open("weights.pkl", "rb"))
model = inception.inception_resnet_v1((160, 160, 3))

for lay in model.layers:

    if re.search("Conv2d", lay.name):

        if not re.search("BatchNorm", lay.name):
            # Convolutional Layer
            if lay.name + "/biases:0" in weights.keys():
                lay.set_weights([weights[lay.name + "/weights:0"], weights[lay.name + "/biases:0"]])
            else:
                lay.set_weights([weights[lay.name + "/weights:0"]])

        else:
            # BatchNorm layer
            beta = weights[lay.name + "/beta:0"]
            moving_mean = weights[lay.name + "/moving_mean:0"]
            moving_var = weights[lay.name + "/moving_variance:0"]
            lay.set_weights([beta, moving_mean, moving_var])

model.save_weights("InceptionResnetV1_weights.h5")
