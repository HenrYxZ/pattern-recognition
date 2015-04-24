from skimage.feature import hog
from skimage import color
from skimage.transform import resize
import numpy as np
import sys

# Returns a list with the histogram concatenation
def get_cc_feats(image, cc_name):
  threshold = 200 
  if len(image.shape) == 3:
    # Using CVL
    height, width, channels = image.shape
    # Use a threshold to get a binary image
    
    bw_img = (color.rgb2gray(image) * 255) < threshold
  else:
    # Using MNIST
    height, width = image.shape
    bw_img = image < threshold

  vertical_step = height / 3
  horizontal_step = width / 3
  boxes = []

  # Define 9 boxes to divide the image (independent of the image resolution)
  for i in range(3):
    for j in range(3):
      vertical_a = i * vertical_step
      vertical_b = (i + 1) * vertical_step - 1
      horizontal_a = j * horizontal_step
      horizontal_b = (j + 1) * horizontal_step - 1
      print("vertical_a = {0}, vertical_b = {1}".format(vertical_a, vertical_b))
      print("horizontal_a = {0}".format(horizontal_a))
      print("horizontal_b = {0}".format(horizontal_b))
      print("bw_img shape = {0}".format(bw_img.shape))
      boxes.append(bw_img[vertical_a : vertical_b][horizontal_a : horizontal_b])

  # Get the histogram for each box and concatenate them
  print ("boxes {0}".format(boxes))
  sys.exit()
  #-----------------------------------------------------------------------------
  img_hist = []
  for box in boxes:
    print ("box {0}".format(box))
    if (cc_name == "4c"):
      local_hist = local_4c_hist(box)
    elif (cc_name == "8c"):
      local_hist = local_8c_hist(box)
    else:
      local_hist = local_mixed_hist(box)
    # List concatenation
    img_hist = img_hist + local_hist
  return img_hist

def local_4c_hist(box):
  directions = ["n", "s", "w", "e"]
  return local_cc_hist(box, directions)

def local_8c_hist(box):
  directions = ["nw", "sw", "ne", "se"]
  return local_cc_hist(box, directions)

def local_mixed_hist(box):
  hist_4c = local_cc_hist(box, ["n", "s", "w", "e"])
  hist_8c = local_cc_hist(box, ["nw", "sw", "ne", "se"])
  return hist_4c + hist_8c

def local_cc_hist(box, directions):
  hist = [0] * 16
  # for each black point do this
  height, width = box.shape
  for i in range(1, height - 1):
    for j in range(1, width - 1):
      if box[i][j] == False:
        # This is a black point
        print ("black point [{0}, {1}]".format(i, j))
        point = [i, j]
        bin_number = 0
        for i in range(len(directions)):
          if hit_4c(box, point, directions[i]):
            print ("hit in {0}".format(directions[i]))
            bin_number += 2 ** i
        hist[bin_number] += 1
  return hist

def hit_4c(box, starting_point, direction):
  current_point = starting_point
  # print ("current_point {0}".format(current_point))
  if direction == "n":
    while in_limits(box, current_point):
      current_point[0] -= 1
      # If we hit something not background
      if current_point == True:
        return True
    return False
  elif direction == "s":
    while in_limits(box, current_point):
      current_point[0] += 1
      if current_point:
        return True
    return False
  elif direction == "w":
    while in_limits(box, current_point):
      current_point[1] -= 1
      if current_point:
        return True
    return False
  else:
    while in_limits(box, current_point):
      current_point[1] += 1
      if current_point:
        return True
    return False

def hit_8c(box, starting_point, direction):
  current_point = starting_point
  if direction == "nw":
    while in_limits(box, current_point):
      current_point[0] -= 1
      current_point[1] -= 1
      # If we hit something not background
      if current_point == True:
        return True
    return False
  elif direction == "sw":
    while in_limits(box, current_point):
      current_point[0] += 1
      current_point[1] -= 1
      if current_point:
        return True
    return False
  elif direction == "ne":
    while in_limits(box, current_point):
      current_point[0] -= 1
      current_point[1] += 1
      if current_point:
        return True
    return False
  else:
    while in_limits(box, current_point):
      current_point[0] += 1
      current_point[1] += 1
      if current_point:
        return True
    return False

def normalize(hist):
  total = sum(hist)
  for i in range(len(hist)):
    hist[i] = hist[i] / total

def in_limits(box, point):
  in_limits_rows = point[0] >= 0 and point[0] < len(box)
  in_limits_cols = point[1] >= 0 and point[1] < len(box[0])
  if in_limits_rows and in_limits_cols:
    return True
  else:
    return False


def get_hog_feats(image, bins, pixels_per_cell_side, cells_per_block_side):
  image_resized = color.rgb2gray(image)
  fd = hog(image_resized, orientations=bins, pixels_per_cell=(pixels_per_cell_side, pixels_per_cell_side),
                        cells_per_block=(cells_per_block_side, cells_per_block_side))
  return fd
