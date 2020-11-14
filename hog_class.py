#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 20:49:59 2020

@author: Waquar Shamsi
"""

################################    HOG CALCULATION     #################################
class my_hog:
  '''


  '''
  def __init__(self):
    pass

  def hog_fun(self,image):
    '''
      Input:  A numpy representation of a 'grayscale' image of ratio(1:1) (with number of pixels=power(2), 
      (if color image is to be used, pre-process it to make it grayscale)
      Returns: A feature extracted image using Histogram of Oriented Gradients
    '''
    hog_image = []
    #takes an image and finds patches hists and then merges them
    num_patches = (image.shape[0]*image.shape[1]) / (8*8)           # Each block is of size 8*8
    num_patches = int(num_patches)
    patches = []
    #later iterate of each block, as of now taking only single block/patch
    for b in range(num_patches):
      row=0
      for _ in range(int(image.shape[0]/8)):
        col=0
        for _ in range(int(image.shape[1]/8)):
          patch = image[row:row+8,col:col+8]
          patch_hist = self.get_patch_hist(patch)
          patches.append(patch_hist)
          col+=8
        row+=8
    
    #### NORMALIZE  USING WINDOW (here blocks of 2x2) #####
    num=0
    for _ in range(int( ( (image.shape[0]/8) -1)* ( (image.shape[1]/8) -1 ) )): 
      block = self.normalize(patches[num][0],patches[num+1][0],patches[num+4][0],patches[num+5][0])

      for i in block:  # ADD FEATURES TO HOG IMAGE
        hog_image.append(i)
    return np.array(hog_image).transpose()

  # create numpy arrays for one patch - 
  def get_patch_hist(self,patch):
    '''
      Input: A single patch of size 8*8 pixels
      Returns: A numpy histogram for a single patch 
    '''
    x_g = self.get_x_gr(patch)
    y_g = self.get_y_gr(patch)
    mags = self.get_mags(x_g,y_g)
    dirs = self.get_dirs(x_g,y_g)
    hist = self.get_hist(mags,dirs)
    return hist

  def get_x_gr(self,patch):
    '''
      Input: A single patch of size 8*8 pixels
      Returns: Gradient matrix for the patch in X direction
    '''
    g_x = np.zeros((8,8))
    for row in range(8):    # FOR EACH ROW
      for col in range(8):  # FOR EACH COLUMN
        if col==7: # IF LAST COLUMN THEN SUBTRACT FROM 0
          g_x[row,col] = 0-patch[row,col-1]                                                                                         # DOING THIS RESULTS IN EDGES, BETTER MAKE IT 0
        elif col==0:              # IF FIRST COLUMN THEN SUBTRACT 0, MEANS SAME VALUE AS RIGHT
          g_x[row,col] = patch[row,col+1]
        else: # FOR ALL OTHER COLUMNS EXCEPT LAST AND FIRST
          g_x[row,col] = patch[row,col+1] - patch[row,col-1] # SUBTRACT FROM RIGHT VALUE TO LEFT VALUE
    return g_x

  def get_y_gr(self,patch):
    '''
      Input: A single patch of size 8*8 pixels
      Returns: Gradient matrix for the patch in Y direction
    '''
    y_x = np.zeros((8,8))
    for col in range(8):    # FOR EACH COLUMN
      for row in range(8):  # FOR EACH ROW
        if row==7: # IF LAST ROW THEN SUBTRACT FROM 0
          y_x[row,col] = 0-patch[row-1,col]                                                                                         # DOING THIS RESULTS IN EDGES, BETTER MAKE IT 0
        elif row==0:              # IF FIRST ROW THEN SUBTRACT 0, MEANS SAME VALUE AS DOWN
          y_x[row,col] = patch[row+1,col]
        else: # FOR ALL OTHER ROWS EXCEPT LAST AND FIRST
          y_x[row,col] = patch[row+1,col] - patch[row-1,col] # SUBTRACT FROM DOWN VALUE TO UP VALUE
    return y_x
    
  def get_mags(self,x_g,y_g):
    '''
      Input:  Two 2d Matrices for X and Y Gradients
      Returns: A 1d Array of Magnitudes
    '''
    mags = np.zeros(8*8)    # CONVERTED TO A 1D ARRAY
    i=0   # ITERATOR FOR mags
    for row in range(8):    # FOR EACH ROW
      for col in range(8): # FOR EACH COLUMN
        mags[i] = math.sqrt(x_g[row,col]**2 + y_g[row,col]**2)    # MAGNITUDE = sqrt(x^2 + y^2)
        i+=1
    return mags

  def get_dirs(self,x_g,y_g):
    '''
      Input:  Two 2d Matrices for X and Y Gradients
      Returns: A 1d Array of Magnitudes
    '''
    dirs = np.zeros(8*8)    # CONVERTED TO A 1D ARRAY
    i=0   # ITERATOR FOR mags
    for row in range(8):    # FOR EACH ROW
      for col in range(8): # FOR EACH COLUMN
        dirs[i] = math.degrees(math.atan(y_g[row,col]/x_g[row,col]))+90    # DIRECTION = atan(y/x)
        i+=1
    return dirs

  def get_hist(self,mags,dirs):
    bin=[0,20,40,60,80,100,120,140,160,180]   #np.histogram takes one less bin, so total bin is still 9
    hist = np.histogram(dirs,bins=bin)
    return hist

  def normalize(self,p1,p2,p3,p4):
    norm = np.zeros(4*9)
    sum=0
    # p = np.concatenate((p1,p2), axis=0)
    # q = np.concatenate((p3,p4), axis=0)
    # r = np.concatenate((p,q), axis=0)
    for i in p1:
      sum += i**2
    for i in p2:
      sum += i**2
    for i in p3:
      sum += i**2
    for i in p4:
      sum += i**2
    sum = math.sqrt(sum) 
    j=0 
    for i in range(9):
      norm[j] = p1[i]/sum
      j+=1
    for i in range(9):
      norm[j] = p2[i]/sum
      j+=1
    for i in range(9):
      norm[j] = p3[i]/sum
      j+=1
    for i in range(9):
      norm[j] = p4[i]/sum
      j+=1
    return norm