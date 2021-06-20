# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 13:16:21 2021

@author: e-default
"""

import tensorflow.keras.backend as K

def accuracy(y_true, y_pred):
  '''
  Compute classification accuracy with a fixed threshold on distances.
  '''
  return K.mean(K.equal(y_true, K.cast(y_pred < 0.2, y_true.dtype)))
