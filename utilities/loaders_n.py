# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 14:48:09 2021

@author: e-default
"""
import numpy as np 
import matplotlib.pyplot as plt
import os
from skimage.color import rgb2gray
import cv2
import gc

error_code ={
  0: 'Argument img_shape must be 3 dimensional. The given img_shape {} is not 3 dimensional',
  1: 'Argument img_shape must be 3 dimensional tuple',
  2: 'Argument all_paths & all_corresp_classes must either be list or np array',
  3: 'You must call either one of the methods below before forming triplets:\n1) group_imgs\n2) load_paths\n3) load_imgs',
  4: 'Argument img_per_class cannot exceed: {}'
  }

class Loader:
  def __init__(self, img_shape=None, img_processor=None):
    # validate img_shape
    if img_shape == None:
      self.img_shape = None
    elif type(img_shape) == tuple:
      if len(img_shape) != 3:
        assert False, error_code[0].format(img_shape)
      self.img_shape = img_shape
    elif type(img_shape) != None:
      assert False, error_code[1]
      
    # create the image preprocessing function
    if img_processor != None:
      self.img_processor = img_processor
      self.__process__ = True
    else:
      self.__process__ = False
      
  def __groupimgs__(self, all_imgs, all_corresp_classes):
    # find all unique values for classes
    self.unique_classes, count_per_class = np.unique(all_corresp_classes, 
                                                return_counts=True)
    
    # get maximum img per class
    self.__max_img_per_class__ = min(count_per_class)
    
    # group all imgs by their labels
    self.imgs_collection = []
    for unique_class in self.unique_classes:
      # create the mask
      mask = np.where(all_corresp_classes == unique_class, True, False)
      selected_imgs = all_imgs[mask]
      self.imgs_collection.append(selected_imgs)
    
    # IMPROVEMENT
    # if self.img_shape == None then only do below
    # else, we reshape all images into the intended shape
    self.img_shape = self.imgs_collection[0][0].shape
      
  def __loadpaths__(self, all_paths, all_corresp_classes):
    # validate the arguments
    args = [all_paths, all_corresp_classes]
    for i, arg in enumerate(args):
      if type(arg) == list:
        args[i] = np.array(arg)
      elif type(arg) != np.ndarray:
        assert False, error_code[2]
        
    self.all_paths = args[0]
    self.all_corresp_classes = args[1]
    
    # get all unique class, and the respective counts
    self.unique_classes, count_per_class = np.unique(self.all_corresp_classes, 
                                                return_counts=True)
    
    # get maximum img per class
    self.__max_img_per_class__ = min(count_per_class)
    
  def __loadimgs__(self, all_paths=None, all_corresp_classes=None, img_size_per_class=None, rearrange=True):
    # get paths and corresponding classes
    if all_paths != None and all_corresp_classes != None:
      self.__loadpaths__(all_paths=all_paths, all_corresp_classes=all_corresp_classes)

    # validate img_size_per_class
    if img_size_per_class != None:
      condition = img_size_per_class <= self.__max_img_per_class__
      assert not condition, error_code[4].format(self.__max_img_per_class__)
    else:
      img_size_per_class = -1
      
    # shuffle the data
    if rearrange:
      np.random.seed(100)
      np.random.shuffle(self.all_paths)
      np.random.seed(100)
      np.random.shuffle(self.all_corresp_classes)
      
    # crete a list of images, where each sublist stores a class of images
    self.imgs_collection = []
    self.classes = []
    for unique_class in self.unique_classes:
      mask = np.where(self.all_corresp_classes == unique_class, True, False)
      paths = self.all_paths[mask]
      self.imgs_collection.append(self.__getimgs__(paths, img_size_per_class=img_size_per_class))
      
  def __getimgs__(self, img_paths, img_size_per_class):
    
    # declare a subfunction to read image
    def read_img(img_path, ch_target):
      # read image
      img = plt.imread(img_path)
      if len(img.shape) == 3 and ch_target == 1:
        img = rgb2gray(img)
      
      # image processing
      if self.__process__:
        img = self.img_processor(img)
      return img
      
    # A) store images in an array
    if self.img_shape != None:
      # declare variables
      r_target, c_target, ch_target = self.img_shape
      qty = len(img_paths)
      if ch_target == 1:
        imgs = np.ones((qty, r_target, c_target)) # will expand dim later
      else:
        imgs = np.ones((qty, r_target, c_target, ch_target))
      # loop
      for i, img_path in enumerate(img_paths):
        if i == img_size_per_class:
          break
        if os.path.isfile(img_path):
          img = read_img(img_path, ch_target=ch_target)
          img = cv2.resize(img, (c_target, r_target))
        imgs[i] = imgs
      # expand dims if required
      if ch_target == 1:
        imgs = np.expand_dims(imgs, axis = -1)
        
      return imgs
            
    # B) store images in a list
    else:
      # declare variables
      imgs = []
      # loop
      for i, img_path in enumerate(img_paths):
        if i == img_size_per_class:
          break
        if os.path.isfile(img_path):
          img = read_img(img_path, ch_target=None)
          if len(img.shape) == 2:
            img = np.expand_dims(img, axis = -1)
        imgs.append(img)
        
      return imgs


class BatchTripletLoader(Loader):
  def __init__(self, img_shape=None, img_processor=None, img_augmentor=None):
    '''
    
    Parameters
    ----------
    img_shape : TYPE, optional
      - must be 3 dimensional tuple
      - (row, col, ch), ch = 1 for grayscale
    img_processor : TYPE, optional
      - if user wish to add certain preprocessing to the images
      - can define a function for image processing
      - then pass into this method as the argument for img_processor
    img_augmentor : TYPE, optional
      - if user wish to add augment the images in a way that ImageDataGenerator
      - can define a function for image augmentation
      - then pass into this method as the argument for img_augmentor
      
    '''
    # call the constructor for Loader
    Loader.__init__(self, img_shape, 
                    img_processor=img_processor)
    
    # create the image augmentation function
    if img_augmentor != None:
      self.img_augmentor = img_augmentor
      self.__augment__ = True
    else:
      self.__augment__ = False
      
    # declare mode as -1 (since havent feed in any data yet)
    self.mode = -1
  
  def group_imgs(self, imgs, labels):
    self.__groupimgs__(imgs, labels)
    self.mode = 0
    
  def load_paths(self, paths, labels):
    self.__loadpaths__(paths, labels)
    self.mode = 1
    
  def load_imgs(self, paths, labels, img_per_class=None, shuffle=True):
    self.__loadimgs__(paths, labels, img_per_class, shuffle)
    self.mode = 2
    
  def get_batch_random(self, img_per_class = 20, shuffle = True):
    # make sure images/paths are preloaded
    assert self.mode != -1, error_code[3]
    
    # load images if self.mode == 1
    if self.mode == 1:
      self.__loadimgs__()
      
    # shuffle the sequence of images in each class
    for idx_class in range(len(self.unique_classes)):
      np.random.shuffle(self.imgs_collection[idx_class])

    # total class
    total_class = len(self.unique_classes)
    # total possible triplet 
    same_pairs_qty_per_class = sum([i for i in range(img_per_class)])
    same_pairs_qty = same_pairs_qty_per_class * total_class
    total_triplets_qty = int(same_pairs_qty)
    # dimension of the pairs
    row, col, ch = self.img_shape
    triplet_dim = (total_triplets_qty, row, col, ch)

    # define some variables
    idx = 0
    self.triplets = [np.ones(triplet_dim, dtype = np.float32),
                     np.ones(triplet_dim, dtype = np.float32),
                     np.ones(triplet_dim, dtype = np.float32)]
    
    # choose a class
    for idx_class_1, imgs_class_1 in enumerate(self.imgs_collection):
      # define all other classes as negative
      all_idx_class_2 = [i for i in range(total_class)]
      all_idx_class_2.remove(idx_class_1)
      np.random.shuffle(all_idx_class_2)
      
      # choose one image in this class as anchor
      for i in range(img_per_class):
        img_1 = imgs_class_1[i]
        if self.__augment__:
          img_1 = self.img_augmentor(img_1)
        
        # choose other image as positive
        for j in range(i + 1, img_per_class, 1):
          img_2 = imgs_class_1[j]
          if self.__augment__:
            img_2 = self.img_augmentor(img_2)
          
          # get a negative
          selected_class = idx % len(all_idx_class_2)
          idx_class_2 = all_idx_class_2[selected_class]
          k = (idx // len(all_idx_class_2)) % self.__max_img_per_class__
          img_3 = self.imgs_collection[idx_class_2][k]
          if self.__augment__:
            img_3 = self.img_augmentor(img_3)
          
          # form triplet
          self.triplets[0][idx] = img_1.copy()
          self.triplets[1][idx] = img_2.copy()
          self.triplets[2][idx] = img_3.copy()
          idx += 1
    
    if shuffle:
      np.random.seed(100)
      np.random.shuffle(self.triplets[0])
      np.random.seed(100)
      np.random.shuffle(self.triplets[1])
      np.random.seed(100)
      np.random.shuffle(self.triplets[2])
      
  def get_batch_semi_hard(self, img_per_class = 20, model = None, hard_margin = 0.8, shuffle = True):
    # make sure images/paths are preloaded
    assert self.mode != -1, error_code[3]
    
    # load images if self.mode == 1
    if self.mode == 1:
      self.__loadimgs__()
      
    # get the randomly formed triplet pairs
    self.get_batch_random(img_per_class=img_per_class, shuffle = False)
    
    # filter out all triplets where the AN distance are lower than hard_margin
    y_pred = model.predict([ self.triplets[0], self.triplets[1], self.triplets[2] ])
    AN = y_pred[:,1]
    mask = np.where(AN < hard_margin, True, False)
    mask = mask.reshape((mask.shape[0]))
    
    # form new triplets
    A = self.triplets[0][mask]
    P = self.triplets[1][mask]
    N = self.triplets[2][mask]
    
    del self.triplets
    gc.collect()
    
    self.triplets = [A, P, N]
    
    if shuffle:
      np.random.seed(100)
      np.random.shuffle(self.triplets[0])
      np.random.seed(100)
      np.random.shuffle(self.triplets[1])
      np.random.seed(100)
      np.random.shuffle(self.triplets[2])
      
  def get_batch_hard(self, img_per_class = 20, model = None, hard_margin = 0.8, shuffle = True):
    # make sure images/paths are preloaded
    assert self.mode != -1, error_code[3]
    
    # load images if self.mode == 1
    if self.mode == 1:
      self.__loadimgs__(self)
      
    # get the randomly formed triplet pairs
    self.get_batch_random(img_per_class=img_per_class, shuffle = False)
    
    # filter out all triplets where the AN distance are lower than hard_margin
    y_pred = model.predict([ self.triplets[0], self.triplets[1], self.triplets[2] ])
    AN = y_pred[:,1]
    mask_1 = np.where(AN < hard_margin, True, False)
    mask_1 = mask_1.reshape((mask_1.shape[0]))
    
    # filter out all triplets where AP distance are higher than the margin
    hard_margin = 1 - hard_margin
    AP = y_pred[:,0]
    mask_2 = np.where(AP > hard_margin, True, False)
    mask_2 = mask_2.reshape((mask_2.shape[0]))
    
    # get the final mask
    mask = mask_1 * mask_2
    
    # form new triplets
    A = self.triplets[0][mask]
    P = self.triplets[1][mask]
    N = self.triplets[2][mask]
    
    del self.triplets
    gc.collect()
    
    self.triplets = [A, P, N]
    
    if shuffle:
      np.random.seed(100)
      np.random.shuffle(self.triplets[0])
      np.random.seed(100)
      np.random.shuffle(self.triplets[1])
      np.random.seed(100)
      np.random.shuffle(self.triplets[2])