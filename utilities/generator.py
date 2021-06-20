# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 21:37:12 2021

@author: e-default
"""

# Outer Class - Inner Class
# People usually initialize inner class in __init__ of Outer Class
# What I did is not really following the convention
# https://www.geeksforgeeks.org/inner-class-in-python/

# Custom Image Data Generator using Tensorflow/Keras
# https://towardsdatascience.com/implementing-custom-data-generators-in-keras-de56f013581c

# Change Spyder Indentation Level
# https://stackoverflow.com/questions/36187784/changing-indentation-settings-in-the-spyder-editor-for-python

import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class Generators:
  '''
  PURPOSE:
  - To define a Class Generators, that hold multiple version of ImageDataGenerator
  
  GUIDES:
  1) during initialization
  - feed a defined ImageDataGenerator instance OR
  - let the __init__ to create an instance of ImageDataGenerator
  
  2) get generator
  - there are 3 METHODS (OOP) to get generator (normal, pair, or triplet generator)
  - depending which type of generator required
  - pass in the original dataset to be augmented
  - the method used will return a generator that gives user the augmented images
  - after each epoch, user can call the on_epoch_end() to create new round of augmented images
  '''
  class PairGenerator(tf.keras.utils.Sequence):
    '''
    PURPOSE:
    - To define a Generator that could form Pairs
    '''
    def __init__(self, input_gen1, input_gen2):

      self.gen1 = input_gen1
      self.gen2 = input_gen2

      assert len(input_gen1) == len(input_gen2)

    def __len__(self):
      # number of bacth in the sequence
      return len(self.gen1)

    def __getitem__(self, index):
      # used to generate one batch of data
      x1 = self.gen1[index]
      x2 = self.gen2[index][0]
      y = self.gen2[index][1]

      assert len(x1) == len(x2) == len(y)

      return [x1, x2], y

    def on_epoch_end(self):
      # this method will be called at the end of every epoch
      self.gen1.on_epoch_end()
      self.gen2.on_epoch_end()
      self.gen2.index_array = self.gen1.index_array
      # print('what is up')

  class TripletGenerator(tf.keras.utils.Sequence):
    '''
    PURPOSE:
    - To define a Generator that could form Triplet
    '''
    def __init__(self, input_gen1, input_gen2, input_gen3):

      self.gen1 = input_gen1
      self.gen2 = input_gen2
      self.gen3 = input_gen3

      assert len(input_gen1) == len(input_gen2) == len(input_gen3)

    def __len__(self):
      # number of bacth in the sequence
      return len(self.gen1)

    def __getitem__(self, index):
      # used to generate one batch of data
      x1 = self.gen1[index]
      x2 = self.gen2[index]
      x3 = self.gen3[index]

      assert len(x1) == len(x2) == len(x3)

      return [x1, x2, x3]

    def on_epoch_end(self):
      # this method will be called at the end of every epoch
      self.gen1.on_epoch_end()
      self.gen2.on_epoch_end()
      self.gen3.on_epoch_end()
      self.gen2.index_array = self.gen1.index_array
      self.gen3.index_array = self.gen1.index_array
      # print('what is up')

  def __init__(self, generator = None):
    '''
    args:
    1) generator
    - ImageDataGenerator to build the PairGenerator and TripletGenerator
    '''
    # Create the base generator
    if type(generator) == tf.keras.preprocessing.image.ImageDataGenerator:
      self.generator = generator
    else:
      self.generator = ImageDataGenerator(horizontal_flip=False, vertical_flip=False, rotation_range=12, brightness_range=[0.5,0.8])

  def get_generator(self, x, y, batch_size = 32, shuffle = True):
    '''
    args:
    1) x
    - images numpy
    2) y
    - labels (class)
    3) batch_size
    - batch size for the generator
    4) shuffle
    - shuffle the dataset

    return 
    - a normal generator
    '''
    gen = self.generator.flow(x = x, y = y, batch_size = batch_size, shuffle = shuffle)
    return gen

  def get_pair_generator(self, X, y, batch_size, shuffle = True):
    '''
    args:
    1) X
    - a pair of images: [images numpy, images numpy]
    - for siamese network
    2) y
    - labels (same or not same)
    3) batch_size
    - batch size for the generator
    4) shuffle
    - shuffle the dataset

    return:
    - a pair generator for siamese network
    '''
    x1, x2 = X
    gen_1 = self.generator.flow(x = x1, batch_size = batch_size, shuffle = shuffle)
    gen_2 = self.generator.flow(x = x2, y = y, batch_size = batch_size, shuffle = shuffle)
    gen = self.PairGenerator(gen_1, gen_2)
    gen.on_epoch_end() # first epoch - sequence wrong

    return gen

  def get_triplet_generator(self, X, batch_size, shuffle = True):
    '''
    args:
    1) X
    - a triplet of images: [images numpy, images numpy, images numpy]
    - for siamese network + triplet loss
    2) batch_size
    - batch size for the generator
    3) shuffle
    - shuffle the dataset

    return 
    - a triplet generator for siamese network + triplet loss
    '''
    x1, x2, x3 = X
    gen_1 = self.generator.flow(x = x1, batch_size = batch_size, shuffle = shuffle)
    gen_2 = self.generator.flow(x = x2, batch_size = batch_size, shuffle = shuffle)
    gen_3 = self.generator.flow(x = x3, batch_size = batch_size, shuffle = shuffle)
    gen = self.TripletGenerator(gen_1, gen_2, gen_3)
    gen.on_epoch_end() # first epoch - sequence wrong
    
    return gen
  