#Import Dependencies
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
%matplotlib inline
import scipy.misc
import pydot

from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input

from keras.initializers import glorot_uniform

from keras import layers
from keras.layers import ZeroPadding2D, Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Add

from keras.models import Model, load_model
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

import keras.backend as K
K.set_image_data_format("channels_last")
K.set_learning_phase(1)

from Ipython.display import SVG
from resnets_utils import *

def identity_block(X, f, filters, stage, block):
  conv_name_base = "res" + str(stage) + block + "_branch"
  bn_name_base = "bn" + str(stage) + block + "_branch"
  
  F1, F2, F3 = filters
  
  X_shortcut = X
  
  X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1, 1), padding = "valid", name = conv_base + "2a", kernel_initializer = glorot_uniform()(X)
  X = BatchNormalization(axis = 3, name = bn_name_base + "2a")(X)
  X = Activation("relu")(X)
  
  X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = "same", name = conv_base + "2b", kernel_initializer = glorot_uniform()(X)
  X = BatchNormalization(axis = 3, name = bn_name_base + "2b")(X)
  X = Activation("relu")(X)
  
  X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = "valid", name = conv_base + "2c", kernel_initializer = glorot_uniform()(X)
  X = BatchNormalization(axis = 3, name = bn_name_base + "2c")(X)
  
  X = Add()([X, X_shortcut])
  X = Activation("relu")(X)
  
  return X

tf.reset_default_graph()
with tf.Session() as sess:
  A_prev = tf.placeholder("float", [3, 4, 4, 6])
  X = np.random.randn(3, 4, 4, 6)
  A = identity_block(X, 2, [2, 4, 6], 1, "a")
  init = tf.global_variables_initializer()
  sess.run(init)
  output = sess.run([A], {A_prev: X, K.learning_phase(): 0})
  print("output = " + str(output[0][1][1][0]))
  
#Convolutional Block
def convolutional_block(X, f, filters, stage, block, s = 2):
  conv_name_base = "res" + str(stage) + block + "_branch"
  bn_name_base = "bn" + str(stage) + block + "_branch"
  
  F1, F2, F3 = filters
  
  X_shortcut = X
  
  X = Conv2D(F1, (1, 1), strides = (s, s), padding = "valid", conv_name_base + "2a", kernel_initializer = glorot_uniform())(X)
  X = BatchNormalization(axis = 3, name = bn_name_base2a + "2a")(X)
  X = Activation("relu")(X)
  
  X = Conv2D(F2, (f, f), strides = (1, 1), padding = "same", conv_name_base + "2b", kernel_initializer = glorot_uniform())(X)
  X = BatchNormalization(axis = 3, bn_name_base + "2b")(X)
  X = Activation("relu")(X)
  
  X = Conv2D(F3, (1, 1), strides = (1, 1), padding = "valid", conv_name_base + "2c", kernel_initializer = glorot_uniform())(X)
  X = BatchNormalization(axis = 3, bn_name_base + "2c")(X)
  
  X_shortcut = Conv2D(F3, (1, 1), strides = (s, s), padding = "valid", conv_name_base + "1", kernel_initializer = glorot_uniform())(X_shortcut)
  X_shortcut = BatchNormalization(axis = 3, bn_name_base + "1")(X_shortcut)
  
  X = Add()([X_shortcut, X])
  X = Activation("relu")(X)
  
  return X

tf.reset_default_graph()
with tf.Session() as sess:
  A_prev = tf.placeholder("float", [3, 4, 4, 6])
  X = np.random.randn(3, 4, 4, 6)
  A = convolutional_block(A_prev, f = 2, filters = [2, 4, 6], 1, "a")
  init = tf.global_variables_initializer()
  sess.run(init)
  output = sess.run([A], {A_prev: X, K.learning_phase(): 0})
  print("output = " + str(output[0][1][1][0]))
  
#Resnet50
def ResNet50(input_shape = (64, 64, 3), classes = 6):
  X_input = Input(input_shape)
  
  X = ZeroPadding2D((3, 3))(x_input)
  
  X = Conv2D((7, 7), filters = 64, strides = (2, 2), name = "conv1", kernel_initializer = glorot_uniform())(X)
  X = BatchNormalization(axis = 3, "bn_conv1")(X)
  X = Activation("relu")(X)
  X = MaxPooling2D((3, 3), stride = (2, 2))(X)
  
  X = convolutional_block(X, f = (3, 3), filters = [64, 64, 256], stage = 2, block = "a", s = 1)
  X = identity_block(X, f = (3, 3), filters = [64, 64, 256], stage = 2, block = "b")
  X = identity_block(X, f = (3, 3), filters = [64, 64, 256], stage = 2, block = "c")
  
  X = convolutional_block(X, f = (3, 3), filters = [128, 128, 512], stage = 3, block = "a", s = 2)
  X = identity_block(X, f = (3, 3), filters = [128, 128, 512], stage = 3, block = "b")
  X = identity_block(X, f = (3, 3), filters = [128, 128, 512], stage = 3, block = "c")
  X = identity_block(X, f = (3, 3), filters = [128, 128, 512, stage = 3, block = "d")
  
  X = convolutional_block(X, f = (3, 3), filters = [256, 256, 1024], stage = 4, block = "a", s = 2)
  X = identity_block(X, f = (3, 3), filters = [256, 256, 1024], stage = 4, block = "b")
  X = identity_block(X, f = (3, 3), filters = [256, 256, 1024], stage = 4, block = "c")
  X = identity_block(X, f = (3, 3), filters = [256, 256, 1024], stage = 4, block = "d")
  X = identity_block(X, f = (3, 3), filters = [256, 256, 1024], stage = 4, block = "e")
  X = identity_block(X, f = (3, 3), filters = [256, 256, 1024], stage = 4, block = "f")
  
  X = convolutional_block(X, f = (3, 3), filters = [512, 512, 2048], stage = 5, block = "a", s = 2)
  X = identity_block(X, f = (3, 3), filters = [256, 256, 2048], stage = 5, block = "b")
  X = identity_block(X, f = (3, 3), filters = [256, 256, 2048], stage = 5, block = "c")
  
  X = AveragePooling2D(pool_size = (2, 2), padding = "same")(X)
  X = Flatten()(X)
  X = Dense(classes, activation = "softmax", name = "fc" + str(classes), kernel_initializer = glorot_uniform())(X)
  
  model = Model(inputs = X_input, outputs = X, name = "ResNet50")
  
  return model
  
model = ResNet50(input_shape = (64, 64, 3), classes = 6)
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

#Load Dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

model.fit(X_train, Y_train, epochs = 2, batch_size = 32)
preds = model.evaluate(X_test, Y_test)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))

model = load_model("ResNet50.h5")
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))

img_path = "images/my_image.jpg"
my_image = scipy.misc.imread(img_path)
imshow(my_image)
img = image.load_img(img_path, target_size = (64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print('Input image shape:', x.shape)
print("class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ")
print(model.predict(x))

model.summary()
plot_model(model, to_file='model.png')
SVG(model_to_dot(model).create(prog='dot', format='svg'))
