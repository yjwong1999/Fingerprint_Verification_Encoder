# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 19:33:01 2021

@author: e-default
"""
import numpy as np 
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import cv2
import gc

error_code ={
  0: 'Argument all_paths & all_corresp_classes must either be list or np array',
  1: 'Argument img_shape must be 3 dimensional tuple',
  2: 'Argument img_shape must be 3 dimensional tuple (row, col, ch).\nThe given img_shape {} is not 3 dimensional',
  3: 'Argument img_per_class cannot exceed: {}',
  4: 'Argument img_per_class cannot smaller than 2',
  5: 'You must call either one of the methods below before forming triplets:\n1) group_imgs\n2) load_paths\n3) load_imgs'
  }

class Loader:
  def __init__(self, img_processor=None):      
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
    
    # define img_shape
    self.img_shape = self.imgs_collection[0][0].shape

  def __loadimgs__(self, all_paths, all_corresp_classes, img_shape, img_size_per_class=None, rearrange=True):      
    # validate the arguments -> all_paths/all_classes
    args = [all_paths, all_corresp_classes]
    for i, arg in enumerate(args):
      if type(arg) == list:
        args[i] = np.array(arg)
      elif type(arg) != np.ndarray:
        assert False, error_code[0]
    all_paths, all_corresp_classes = args
    
    # validate img_shape
    if type(img_shape) != tuple:
      assert False, error_code[1]
    else:
      if len(img_shape) != 3:
        assert False, error_code[2].format(img_shape)
      self.img_shape = img_shape

    # get all unique class, and the respective counts
    self.unique_classes, count_per_class = np.unique(all_corresp_classes, 
                                                return_counts=True)
    
    # get maximum img per class
    self.__max_img_per_class__ = min(count_per_class)

    # validate img_size_per_class
    if img_size_per_class != None:
      condition = img_size_per_class <= self.__max_img_per_class__
      assert condition, error_code[3].format(self.__max_img_per_class__)
      condition = img_size_per_class >= 2
      assert condition, error_code[4]
    else:
      img_size_per_class = -1
      
    # shuffle the data
    if rearrange:
      np.random.seed(100)
      np.random.shuffle(all_paths)
      np.random.seed(100)
      np.random.shuffle(all_corresp_classes)
      
    # crete a list of images, where each sublist stores a class of images
    self.imgs_collection = []
    for unique_class in self.unique_classes:
      mask = np.where(all_corresp_classes == unique_class, True, False)
      paths = all_paths[mask]
      self.imgs_collection.append(self.__getimgs(paths, img_size_per_class=img_size_per_class))
      
  def __getimgs(self, img_paths, img_size_per_class):
    
    # declare a subfunction to read image
    def read_img(img_path):
      # read image
      img = plt.imread(img_path)
      if len(img.shape) == 3 and self.img_shape[2] == 1:
        img = rgb2gray(img)
      
      # image processing
      if self.__process__:
        img = self.img_processor(img)
        
      return img
      
    # store images in an array
    r_target, c_target, ch_target = self.img_shape
    if img_size_per_class == -1:
      qty = len(img_paths)
    else:
      qty = img_size_per_class
    if ch_target == 1:
      imgs = np.ones((qty, r_target, c_target)) # will expand dim later
    else:
      imgs = np.ones((qty, r_target, c_target, ch_target))
    
    # loop
    for i, img_path in enumerate(img_paths):
      if i == img_size_per_class:
        break
      img = read_img(img_path)
      img = cv2.resize(img, (c_target, r_target))
      imgs[i] = img
    
    # expand dims if required
    if ch_target == 1:
      imgs = np.expand_dims(imgs, axis = -1)
        
    return imgs
  
class AnchorTripletLoader(Loader):
  def __init__(self, img_processor=None, img_augmentor=None):
    '''
    
    Parameters
    ----------
    img_processor : TYPE, optional
      - if user wish to add certain preprocessing to the images
      - can define a function for image processing
      - then pass into this method as the argument for img_processor
    img_augmentor : TYPE, optional
      - if user wish to add augment the images in a way that ImageDataGenerator
      - can define a function for image augmentation
      - then pass into this method as the argument for img_augmentor
      
    Remarks
    ----------
    - img_processor is applied straight after reading the images
    - img_augmentor is applied when forming Triplets
    
    '''
    # call the constructor for Loader
    Loader.__init__(self, img_processor=img_processor)  
    
    # create the image augmentation function
    if img_augmentor != None:
      self.img_augmentor = img_augmentor
      self.__augment__ = True
    else:
      self.__augment__ = False
      
    # declare mode as -1 (since havent feed in any data yet)
    self.mode = -1
    
  def load_imgs(self, anchor_paths, anchor_labels, normal_paths, normal_labels,
                img_shape, img_per_class=None, shuffle=True):
    # get anchor
    self.__loadimgs__(anchor_paths, anchor_labels, img_shape, img_per_class, shuffle)
    self.anchors_collection = self.imgs_collection.copy()
    self.anchors_classes = self.unique_classes.copy()
    del self.imgs_collection
    del self.unique_classes
    gc.collect()
    
    # get other images
    self.__loadimgs__(normal_paths, normal_labels, img_shape, img_per_class, shuffle)
    
    # make sure the order of class is the same
    same = np.product(self.anchors_classes == self.unique_classes)
    assert bool(same), "Please add error code! Sequence of class for anchor and normal images are diff!"
    
    # declare mode
    self.mode = 1
    
  def load_paths(self, anchor_paths, anchor_labels, normal_paths, normal_labels,
                img_shape):
    # validate the arguments -> all_paths/all_classes
    args = [anchor_paths, anchor_labels, normal_paths, normal_labels]
    for i, arg in enumerate(args):
      if type(arg) == list:
        args[i] = np.array(arg)
      elif type(arg) != np.ndarray:
        assert False, error_code[0]
    self.anchor_paths, self.anchor_labels, self.normal_paths, self.normal_labels = args
  
    # validate img_shape
    if type(img_shape) != tuple:
      assert False, error_code[1]
    else:
      if len(img_shape) != 3:
        assert False, error_code[2].format(img_shape)
      self.img_shape = img_shape
    
    # get anchor images and labels
    self.__loadimgs__(self.anchor_paths, self.anchor_labels, img_shape)
    self.anchors_collection = self.imgs_collection.copy()
    self.anchors_classes = self.unique_classes.copy()
    del self.imgs_collection
    del self.unique_classes
    gc.collect()
    
    # declare mode
    self.mode = 2
    
  def get_batch_random(self, img_per_class, shuffle = True):
    # make sure images/paths are preloaded
    assert self.mode != -1, error_code[5]
    
    # load images if self.mode == 2
    if self.mode == 2:
      self.__loadimgs__(self.paths, self.labels, self.img_shape, img_per_class, shuffle)

      # make sure the order of class is the same
      same = np.product(self.anchors_classes == self.unique_classes)
      assert bool(same), "Please add error code! Sequence of class for anchor and normal images are diff!"
      
    # shuffle the sequence of images in each class
    for idx_class in range(len(self.unique_classes)):
      np.random.shuffle(self.imgs_collection[idx_class])

    # total class
    total_class = len(self.unique_classes)
    # total possible triplet 
    total_triplets_qty = int(total_class * img_per_class)
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
      
      # get amchor
      img_1 = self.anchors_collection[idx_class_1][-1]
      if self.__augment__:
        img_1 = self.img_augmentor(img_1)    
      
      # get positive
      for i in range(img_per_class):
        img_2 = imgs_class_1[i]
        if self.__augment__:
          img_2 = self.img_augmentor(img_2)      
          
        # get a negative
        selected_class = idx % len(all_idx_class_2)
        idx_class_2 = all_idx_class_2[selected_class]
        k = (idx // len(all_idx_class_2)) % img_per_class
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
    
