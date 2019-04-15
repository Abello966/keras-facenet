# keras-facenet
This repository uses davidsandberg's [facenet repository](https://github.com/davidsandberg/facenet) to turn the pre-trained Tensorflow models available in Keras

Many thanks to the answers on this Stack Overflow's [question](https://stackoverflow.com/questions/44466066/how-can-i-convert-a-trained-tensorflow-model-to-keras)


# How to Use
 - Go to the original repository
 - Get one of the pre-trained models in TF form
 - Run python3 extract\_tf\_weights.py \<path\_to\_model\> to generate weights\.pkl, the file containing the weights in numpy array form
 - Run python3 load\tf\_keras.py to generate the model in Keras format
 - Enjoy the model in Keras!


This was tested on Linux with tensorflow 1.4 to 1.12, and is probably going to end up obsolete real soon. I have no intention of updating this repository to address
intricacies of new versions. PRs welcome!
