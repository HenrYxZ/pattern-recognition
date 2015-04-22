from skimage.feature import hog
from skimage import color
from skimage.transform import resize

import numpy as np

def get_4c_feats(image):
  pass

def get_8c_feats(image):
  pass

def get_mixed_feats(image):
  pass

def get_hog_feats(image, bins, pixels_per_cell_side, cells_per_block_side):
  image_resized = resize(color.rgb2gray(image),(64,64))
  fd = hog(image_resized, orientations=bins, pixels_per_cell=(pixels_per_cell_side, pixels_per_cell_side),
                        cells_per_block=(cells_per_block_side, cells_per_block_side))
  return fd