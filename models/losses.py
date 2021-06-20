# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:52:43 2021

@author: e-default

FUTURE IMRPOVEMENT:
1) Add Triplet Loss
"""
# Custom Layers- Contrastive Loss
# https://www.coursera.org/lecture/custom-models-layers-loss-functions-with-tensorflow/coding-contrastive-loss-SzfUz

# Change Spyder Indentation Level
# https://stackoverflow.com/questions/36187784/changing-indentation-settings-in-the-spyder-editor-for-python

from tensorflow.keras.losses import Loss
import tensorflow.keras.backend as K

class ContrastiveLoss(Loss):
  margin = 0
  def __init__(self, margin):
    '''
    args:
    1) margin:
    - the margin needed in the equation
    '''
    super().__init__()
    self.margin = margin

  def call(self, y_true, y_pred):
    '''
    args:
    1) y_true:
    - 1 if same, 0 if diff
    2) y_pred
    - the distance between the image pair
    - small if same, big if different
    '''
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(self.margin - y_pred, 0))
    loss = K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    return loss
  
class ModifiedTripletLoss(Loss):
  def __init__(self, scale=2, thresh=0.6, margin=0.9):
    '''
    args:
    1) scale:
    - to scale up the penalization
    '''
    super().__init__()
    self.scale = scale
    self.thresh = thresh
    self.margin = margin
    
  def call(self, y_true, y_pred):
    '''
    args:
    1) y_true:
    - None
    2) y_pred
    - consists of AP and AN
    - AP should be small, AN should be big
    '''

    AP = y_pred[:,0]
    AN = y_pred[:,1]

    base_loss_a = self.scale * (K.square(AP - 0.5) + K.square(AN - 0.5)) + 1
    base_loss_b = K.maximum(AP + self.margin - AN, 0)

    coef_1 = -1.0 * K.minimum(AP - self.thresh, 0.0) / K.maximum(-1.0 * K.minimum(AP - self.thresh, 0.0), K.epsilon())
    coef_2 = K.maximum(AN - (1.0 - self.thresh), 0.0) / K.maximum(K.maximum(AN - (1.0 - self.thresh), 0.0), K.epsilon())

    loss =  base_loss_a + coef_1 * coef_2 * (base_loss_b - base_loss_a)

    return loss

class NewTripletLoss(Loss):
  scale = 0
  def __init__(self, scale, thresh=0.2):
    '''
    args:
    1) scale:
    - to scale up the penalization
    '''
    super().__init__()
    self.scale = scale
    self.thresh = thresh
    
  def call(self, y_true, y_pred):
    '''
    args:
    1) y_true:
    - None
    2) y_pred
    - consists of AP and AN
    - AP should be small, AN should be big
    '''
    AP = y_pred[:,0]
    AN = y_pred[:,1]
    
    base_loss = self.scale* (K.square(AP) + K.square(AN - 1))
    
    coef_1a = -1.0 * K.minimum(AP - self.thresh, 0.0) / K.maximum(-1.0 * K.minimum(AP - self.thresh, 0.0), K.epsilon())
    coef_1b = -1.0 * K.minimum(AN - self.thresh, 0.0) / K.maximum(-1.0 * K.minimum(AN - self.thresh, 0.0), K.epsilon())
    loss_1 = K.square(AP - 1.0) - K.square(AP)
    
    coef_2a = K.maximum(AP - (1.0 - self.thresh), 0.0) / K.maximum(K.maximum(AP - (1.0 - self.thresh), 0.0), K.epsilon())
    coef_2b = K.maximum(AN - (1.0 - self.thresh), 0.0) / K.maximum(K.maximum(AN - (1.0 - self.thresh), 0.0), K.epsilon())
    loss_2 = K.square(AN) - K.square(AN - 1.0)
    
    
    loss = base_loss + coef_1a*coef_1b*loss_1 + coef_2a*coef_2b*loss_2
    
    return loss
  
class TripletLoss(Loss):
  scale = 0
  def __init__(self, margin):
    '''
    args:
    1) scale:
    - to scale up the penalization
    '''
    super().__init__()
    self.margin = margin
    
  def call(self, y_true, y_pred):
    '''
    args:
    1) y_true:
    - None
    2) y_pred
    - consists of AP and AN
    - AP should be small, AN should be big
    '''
    AP = y_pred[:,0]
    AN = y_pred[:,1]
    basic_loss = AP - AN + self.margin
    loss = K.maximum(basic_loss, 0.0)
    return loss