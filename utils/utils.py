# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 21:16:43 2021

@author: e-default
"""
import numpy as np 
import math

import matplotlib.pyplot as plt

error_code ={
  0: 'Argument all_paths & all_coresp_classes must either be list or np array',
  1: 'Argument fold_num must be integer, which is greater or equal to 2',
  2: 'Argument fold_num must be less than or equal to the total class size ({} in this case), and should be reasonable',
  3: 'Argument min_img_per_class must be integer, which is greater or equal to 2'
  }

# Split the dataset into multiple folds
class KFoldSplit:
  def __init__(self, all_paths, all_coresp_classes, 
               fold_num, min_img_per_class = 2):
    # validate the arguments -> all_paths/all_coresp_classes
    args = [all_paths, all_coresp_classes]
    for i, arg in enumerate(args):
      if type(arg) == list:
        args[i] = np.array(arg)
      elif type(arg) != np.ndarray:
        assert False, error_code[0]
    all_paths, all_corresp_classes = args

    # validate fold_num (must be at least 2)
    assert type(fold_num) == int, error_code[1]
    assert fold_num >= 2, error_code[1]
    
    # make sure min_img_per_class is at least 2
    assert type(min_img_per_class) == int, error_code[3]
    assert min_img_per_class >= 2, error_code[3]
    
    # declare variables
    self.fold_num = fold_num
    self.folds = []
    self.fold_idx = 0
    self.unwanted_classes = []
    self.total_class = 0
    
    # sort the two arrays
    idx = np.argsort(all_coresp_classes)
    all_paths = all_paths[idx]
    all_coresp_classes = all_coresp_classes[idx]    
    
    # get the unique classes name
    unique_classes = np.unique(all_coresp_classes)
        
    # remove fingerprint classes without sufficient image size
    unwanted_classes = []
    for unique_class in unique_classes:
      mask = np.where(all_coresp_classes == unique_class)
      if len(mask[-1]) < min_img_per_class:
        unwanted_classes.append(unique_class)
    self.unwanted_classes = unwanted_classes
    
    if len(unwanted_classes) > 0:
      msg = 'The following classe(s) are removed because it/they do(es) not fulfill the min_img_per_class:\n'
      msg += '{}'.format(unwanted_classes)
      print(msg)
      for unwanted_class in unwanted_classes:
        # remove image paths and the corresponding class labels
        mask = np.where(all_coresp_classes==unwanted_class)
        all_paths = np.delete(all_paths, mask[-1])
        all_coresp_classes = np.delete(all_coresp_classes, mask[-1])
        # remove unwanted_class from unique_classes
        unique_classes = unique_classes[unique_classes!=unwanted_class]
      
    # update total class size
    total_class = len(unique_classes)
    self.total_class = total_class
    
    # validate if the fold_num
    assert fold_num <= total_class, error_code[2].format(total_class)
    
    # get each fold
    fold_size = math.ceil(total_class / self.fold_num)
    for fold_idx in range(self.fold_num):
      # get starting and ending index -> to select class
      start_idx = fold_idx * fold_size
      end_idx = start_idx + fold_size
      end_idx = end_idx if end_idx <= total_class else total_class
    
      # get the training and validation out from all_paths & all_coresp_classes
      start_idx = np.where(all_coresp_classes == unique_classes[start_idx])[0][0]
      end_idx = np.where(all_coresp_classes == unique_classes[end_idx - 1])[0][-1] + 1
      fold  = (all_paths[start_idx:end_idx], all_coresp_classes[start_idx:end_idx])
      self.folds.append(fold)
      
  def get_train_test_split(self):
    # get the split
    all_folds = self.folds.copy()
    cache = [[],[]]
    test_fold = all_folds.pop(self.fold_idx)
    for paths, labels in all_folds:
      cache[0].append(paths)
      cache[1].append(labels)
    train_fold = [np.concatenate(cache[0]), np.concatenate(cache[1])]
    
    # update self.fold_idx for next round
    self.fold_idx += 1
    if self.fold_idx >= self.fold_num:
      self.fold_idx = 0
      print('This is the last fold! If you wish to continue, then you will be reusing the folds!')
      
    # return the split
    return train_fold, test_fold
  
# to show/display images
def show_images(imgs):
  if type(imgs) == list:
    for img in imgs:
      if len(img.shape) == 3:
        r, c, _ = img.shape
        img = np.reshape(img, (r, c))
      plt.imshow(img, cmap = 'gray')
      plt.show()
  else:
    if len(imgs.shape) == 3:
      r, c, _ = imgs.shape
      imgs = np.reshape(imgs, (r, c))
    plt.imshow(imgs, cmap = 'gray')
    plt.show()