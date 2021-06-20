# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 11:43:23 2021

@author: e-default
"""
from tensorflow.keras.layers import Input, Flatten, Lambda, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

# how to define custom models
# https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/

# check if a variable exist
# https://www.kite.com/python/answers/how-to-check-if-a-variable-exists-in-python#:~:text=Check%20if%20a%20variable%20exists%20locally%20or%20globally,a%20variable%20is%20defined%20locally.&text=Use%20globals()%20to%20return%20a%20dictionary%20for%20variables%20defined%20globally.

def cosine_distance(vects):
  '''
  PURPOSE:
  to create a layer to measure the cosine distance
  
  args:
  1) vects
  - contains two tensors
  - the tensors are the encoding 1 and 2 respectively
  '''
  x, y = vects

  xy_dot = K.sum(x * y, axis = 1, keepdims = True)
  x_mag = K.sqrt(K.sum(K.square(x), axis = 1, keepdims = True))
  y_mag = K.sqrt(K.sum(K.square(y), axis = 1, keepdims = True))

  cos_sim = xy_dot / K.maximum((x_mag * y_mag), K.epsilon())
  cos_dist = 1 - cos_sim

  return cos_dist

def euclidean_distance(vects):
  '''
  PURPOSE:
  to create a layer to measure the cosine distance
  
  args:
  1) vects
  - contains two tensors
  - the tensors are the encoding 1 and 2 respectively
  '''
  x, y = vects
  sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
  eucl_dist =  K.sqrt(K.maximum(sum_square, K.epsilon()))
  
  return eucl_dist
  
def get_softmax_model(encoder, classifier):
  '''
  PURPOSE:
  - To create a typical softmax model
  - softmax model = encoder (CNN) + Classifier (Dense/MLP)
  
  args:
  1) encoder
  - any form of encoder is welcomed
  - as long as it is CNN
  2) classifier
  - any form of Multilayer Perceptron (MLP)
  '''
  # get input shape
  input_shape = encoder.input.shape[1:]
  
  # define the architecture
  model_input = Input(shape=input_shape)
  x = encoder(model_input)
  model_output = classifier(x)
  
  # create the model
  model = Model(model_input, model_output)
  
  return model

def get_siamese_network(encoder, **kwargs):
  '''
  PURPOSE:
  - To create a siamese network
  - 2 form of siamese network could be created:
    a) binary classification
    b) distance-based (recommended)
  
  args:
  1) encoder
  - any form of encoder is welcomed
  - as long as it is CNN
  2) **kwargs
  - so far the only argument accepted is MLP
  - which is the defined MLP model
  '''
  # check if user give a MLP layer
  for key, value in kwargs.items():
    if key == 'MLP':
      MLP = value
    else:
      assert False, 'Your **kwargs could only be "MLP"!'
      
  # get input shape
  input_shape = encoder.input.shape[1:]
  
  # feed the image pairs to the encoder
  input_1 = Input(shape=input_shape)
  encoding_1 = encoder(input_1)
  
  input_2 = Input(shape=input_shape)
  encoding_2 = encoder(input_2)

  # flatten the tensors
  encoding_1 = Flatten()(encoding_1)
  encoding_2 = Flatten()(encoding_2)
    
  # insert a MLP layer if required
  if 'MLP' in locals():    
    # feed the encoding to MLP
    encoding_1 = MLP(encoding_1)
    encoding_2 = MLP(encoding_2)
    
    # normalize the feature vectors of the encodings
    l2_norm_layer = Lambda(lambda tensors:K.l2_normalize(tensors))
    encoding_1 = l2_norm_layer(encoding_1)
    encoding_2 = l2_norm_layer(encoding_2)
    
    dist = Lambda(cosine_distance)([encoding_1, encoding_2])
    
    model = Model([input_1, input_2], dist)
    
    return model
    
  else:
    # normalize the feature vectors of the encodings
    l2_norm_layer = Lambda(lambda tensors:K.l2_normalize(tensors))
    encoding_1 = l2_norm_layer(encoding_1)
    encoding_2 = l2_norm_layer(encoding_2)
    
    # get the absolute difference between the two encodings
    L1_dist = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    dist = L1_dist([encoding_1, encoding_2])
    
    # dense layer
    x = Dense(128,activation='relu')(dist)
    x = Dropout(0.2)(x)
    output = Dense(1,activation='sigmoid')(x)
    
    model = Model([input_1, input_2], output)
    
    return model
    
def get_triplet_network(encoder, MLP):
  '''
  PURPOSE:
  - To create a triplet network
  
  args:
  1) encoder
  - any form of encoder is welcomed
  - as long as it is CNN
  2) MLP
  - any form of Multilayer Perceptron (MLP)
  '''
  # get input shape
  input_shape = encoder.input.shape[1:]
  
  # feed the image triplet to the encoder
  input_1 = Input(shape=input_shape)
  encoding_1 = encoder(input_1)
  
  input_2 = Input(shape=input_shape)
  encoding_2 = encoder(input_2)
  
  input_3 = Input(shape=input_shape)
  encoding_3 = encoder(input_3)

  # flatten the tensors
  encoding_1 = Flatten()(encoding_1)
  encoding_2 = Flatten()(encoding_2)
  encoding_3 = Flatten()(encoding_3)
  
  # feed the encodings to MLP
  encoding_1 = MLP(encoding_1)
  encoding_2 = MLP(encoding_2)
  encoding_3 = MLP(encoding_3)
  
  # normalize the feature vectors of the encodings
  l2_norm_layer = Lambda(lambda tensors:K.l2_normalize(tensors))
  encoding_1 = l2_norm_layer(encoding_1)
  encoding_2 = l2_norm_layer(encoding_2)
  encoding_3 = l2_norm_layer(encoding_3)
  
  # get AP, AN
  AP = Lambda(cosine_distance)([encoding_1, encoding_2])
  AN = Lambda(cosine_distance)([encoding_1, encoding_3])
  
  # Concatenate
  output = Concatenate(axis=1)([AP, AN])
  
  # build the model
  model = Model([input_1, input_2, input_3], output)
  
  return model
  
  