import keras as kr
import keras.backend as K

from keras import losses
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, Lambda, Conv2DTranspose, Add, Concatenate
from keras.layers import Input, BatchNormalization, MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU


# Most of the convolutional layers have this structure
def irConv2D(inputs, dimension, size, name="", padding="same", stride=1, train_bn=False):
    conv = Conv2D(dimension, size, name=name, use_bias=False, activation=None, padding=padding, strides=stride)(inputs)
    batch = BatchNormalization(name=name + "/BatchNorm", scale=False)(conv, training=train_bn)
    relu = Activation("relu")(batch)

    return relu

    
#Inception Resnet A
def block35(inputs, scope="", scale=1.0, activation=True, train_bn=False):
    #Branch 0
    tower_conv = irConv2D(inputs, 32, 1, name=scope + "/Branch_0/Conv2d_1x1", train_bn=train_bn)
    
    #Branch 1
    tower_conv1_0 = irConv2D(inputs, 32, 1, name=scope + "/Branch_1/Conv2d_0a_1x1", train_bn=train_bn)
    tower_conv1_1 = irConv2D(tower_conv1_0, 32, 3, name=scope + "/Branch_1/Conv2d_0b_3x3", train_bn=train_bn)

    #Branch 2
    tower_conv2_0 = irConv2D(inputs, 32, 1, name=scope + "/Branch_2/Conv2d_0a_1x1", train_bn=train_bn)
    tower_conv2_1 = irConv2D(tower_conv2_0, 32, 3, name=scope + "/Branch_2/Conv2d_0b_3x3", train_bn=train_bn)
    tower_conv2_2 = irConv2D(tower_conv2_1, 32, 3, name=scope + "/Branch_2/Conv2d_0c_3x3", train_bn=train_bn)

    mixed = Concatenate()([tower_conv, tower_conv1_1, tower_conv2_2])

    updim = Conv2D(K.int_shape(inputs)[3], 1, name=scope + "/Conv2d_1x1", padding='same')(mixed)

    scaled = Lambda(lambda x: x * scale)(updim)

    out = Add()([inputs, scaled])

    if activation:
        out = Activation("relu")(out)

    return out


def block17(inputs, scope="", scale=1.0, activation=True, train_bn=False):
    #Branch 0
    tower_conv = irConv2D(inputs, 128, 1, name=scope + "/Branch_0/Conv2d_1x1", train_bn=train_bn)
    
    #Branch 1
    tower_conv1_0 = irConv2D(inputs, 128, 1, name=scope + "/Branch_1/Conv2d_0a_1x1", train_bn=train_bn)
    tower_conv1_1 = irConv2D(tower_conv1_0, 128, (1, 7), name=scope + "/Branch_1/Conv2d_0b_1x7", train_bn=train_bn)
    tower_conv1_2 = irConv2D(tower_conv1_1, 128, (7, 1), name=scope + "/Branch_1/Conv2d_0c_7x1", train_bn=train_bn)

    mixed = Concatenate()([tower_conv, tower_conv1_2])

    updim = Conv2D(K.int_shape(inputs)[3], 1, name=scope + "/Conv2d_1x1", padding='same')(mixed)

    scaled = Lambda(lambda x: x * scale)(updim)

    out = Add()([inputs, scaled])
    
    if activation:
        out = Activation("relu")(out)

    return out


def block8(inputs, scope="", scale=1.0, activation=True, train_bn=False):
    #Branch 0 
    tower_conv = irConv2D(inputs, 192, 1, name=scope + "/Branch_0/Conv2d_1x1", train_bn=train_bn)
    
    #Branch 1
    tower_conv1_0 = irConv2D(inputs, 192, 1, name=scope + "/Branch_1/Conv2d_0a_1x1", train_bn=train_bn)
    tower_conv1_1 = irConv2D(tower_conv1_0, 192, (1, 3), name=scope + "/Branch_1/Conv2d_0b_1x3", train_bn=train_bn)
    tower_conv1_2 = irConv2D(tower_conv1_1, 192, (3, 1), name=scope + "/Branch_1/Conv2d_0c_3x1", train_bn=train_bn)

    mixed = Concatenate()([tower_conv, tower_conv1_2])

    updim = Conv2D(K.int_shape(inputs)[3], 1, name=scope + "/Conv2d_1x1", padding='same')(mixed)

    scaled = Lambda(lambda x: x * scale)(updim)

    out = Add()([inputs, scaled])

    if activation:
        out = Activation("relu")(out)

    return out


def reduction_a(inputs, scope="", train_bn=False):
    #Branch_0
    tower_conv = irConv2D(inputs, 384, 3, stride=2, padding="valid", name=scope + "/Branch_0/Conv2d_1a_3x3", train_bn=train_bn)

    #Branch_1
    tower_conv1_0 = irConv2D(inputs, 192, 1, name=scope + "/Branch_1/Conv2d_0a_1x1", train_bn=train_bn)
    tower_conv1_1 = irConv2D(tower_conv1_0, 192, 3, name=scope + "/Branch_1/Conv2d_0b_3x3", train_bn=train_bn)
    tower_conv1_2 = irConv2D(tower_conv1_1, 256, 3, stride=2, padding="valid", name=scope + "/Branch_1/Conv2d_1a_3x3", train_bn=train_bn)
    
    #Branch_2
    pool = MaxPooling2D(3, strides=2, padding="valid", name=scope + "MaxPool_1a_3x3")(inputs)

    outs = Concatenate()([tower_conv, tower_conv1_2, pool])
    
    return outs


def reduction_b(inputs, scope="", train_bn=False):
    #Branch_0    
    tower_conv = irConv2D(inputs, 256, 1, name=scope + "/Branch_0/Conv2d_0a_1x1", train_bn=train_bn)
    tower_conv_1 = irConv2D(tower_conv, 384, 3, stride=2, padding="valid", name=scope + "/Branch_0/Conv2d_1a_3x3", train_bn=train_bn)

    #Branch_1
    tower_conv1 = irConv2D(inputs, 256, 1, name=scope + "/Branch_1/Conv2d_0a_1x1", train_bn=train_bn)
    tower_conv1_1 = irConv2D(tower_conv1, 256, 3, stride=2, padding="valid", name=scope + "/Branch_1/Conv2d_1a_3x3", train_bn=train_bn)

    #Branch_2
    tower_conv2 = irConv2D(inputs, 256, 1, name = scope + "/Branch_2/Conv2d_0a_1x1", train_bn=train_bn)
    tower_conv2_1 = irConv2D(tower_conv2, 256, 3, name=scope + "/Branch_2/Conv2d_0b_3x3", train_bn=train_bn)
    tower_conv2_2 = irConv2D(tower_conv2_1, 256, 3, stride=2, padding="valid", name=scope + "/Branch_2/Conv2d_1a_3x3", train_bn=train_bn)

    #Branch_3
    pool = MaxPooling2D(3, strides=2, padding="valid", name=scope + "MaxPool_1a_3x3")(inputs)

    outs = Concatenate()([tower_conv_1, tower_conv1_1, tower_conv2_2, pool])
    return outs

    
def inception_resnet_v1(input_shape, scope="InceptionResnetV1", dropout_prob=0.8, bottle_size=512, train_bn=False):
    incoming = Input(shape=input_shape)

    conv1a = irConv2D(incoming, 32, 3, stride=2, padding="valid", name=scope + "/Conv2d_1a_3x3", train_bn=train_bn)
    
    conv2a = irConv2D(conv1a, 32, 3, padding="valid", name=scope + "/Conv2d_2a_3x3", train_bn=train_bn)
    conv2b = irConv2D(conv2a, 64, 3, name=scope + "/Conv2d_2b_3x3", train_bn=train_bn)
   
    pool3a = MaxPooling2D(3, strides=2, padding='valid', name=scope + "/MaxPool_3a_3x3")(conv2b)

    conv3b = irConv2D(pool3a, 80, 1, padding='valid', name=scope + "/Conv2d_3b_1x1", train_bn=train_bn)
    conv4a = irConv2D(conv3b, 192, 3, padding='valid', name = scope + "/Conv2d_4a_3x3", train_bn=train_bn)
    conv4b = irConv2D(conv4a, 256, 3, stride=2, padding='valid', name = scope + "/Conv2d_4b_3x3", train_bn=train_bn)


    #5x Inception-resnet-a
    block = block35(conv4b, scope=scope +"/Repeat/block35_1", scale=0.17, train_bn=train_bn)
    for i in range(2, 6):
        block = block35(block, scope=scope + "/Repeat/block35_" + str(i), scale=0.17, train_bn=train_bn)
    
    # Reduction A
    reduced = reduction_a(block, scope=scope + "/Mixed_6a", train_bn=train_bn)

    #10x Inception-resnet-b
    block = block17(reduced, scope=scope + "/Repeat_1/block17_1", scale=0.10, train_bn=train_bn)
    for i in range(2, 11):
        block = block17(block, scope= scope + "/Repeat_1/block17_" + str(i), scale=0.10, train_bn=train_bn)

    #Reduction B
    reduced_2 = reduction_b(block, scope=scope + "/Mixed_7a", train_bn=train_bn)

    #5x Inception-Resnet-C
    block = block8(reduced_2, scope=scope + "/Repeat_2/block8_1", scale=0.2, train_bn=train_bn)
    for i in range(2, 6):
        block = block8(block, scope=scope + "/Repeat_2/block8_" + str(i), scale=0.2, train_bn=train_bn)

    #Final block
    block = block8(block, scope=scope + "/Block8", activation=False, train_bn=train_bn)    
        
    #Logits
    avg = AveragePooling2D(K.int_shape(block)[1:3], padding="valid")(block)
    flat = Flatten()(avg)
    drop = Dropout(dropout_prob)(flat) 

    # Bottleneck layer
    btn = Dense(bottle_size, name=scope +"/Bottleneck")(drop)
    btn_bn = BatchNormalization(name=scope + "/Bottleneck/BatchNorm", scale=False)(btn, training=train_bn)

    return Model(inputs=incoming, outputs=btn_bn)

