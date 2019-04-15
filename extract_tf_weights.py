import sys
import re
import pickle as pkl
import facenet.src.facenet as facenet
import tensorflow as tf

path = sys.argv[1] 

sess = tf.Session()
sess.__enter__()
facenet.load_model(path)

init = tf.global_variables_initializer()
sess.run(init)

vars_globals = tf.global_variables()

model_dict = dict()
for var in vars_globals:
    if var.name.startswith("InceptionResnetV1") and not re.search("Adam", var.name):
        try:
            model_dict[var.name] = var.eval()
        except:
            print("Couldn't get {}".format(var.name))

pkl.dump(model_dict, open("weights.pkl", "wb"))
