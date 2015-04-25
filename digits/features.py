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
      boxes.append(bw_img[vertical_a : vertical_b, horizontal_a : horizontal_b])

  img_hist = []
  for box in boxes:
    print ("box = {0}".format(box))
    print ("box shape = {0}".format(box.shape))
    if (cc_name == "4c"):
      local_hist = local_cc_hist(box, "4c")
    elif (cc_name == "8c"):
      local_hist = local_cc_hist(box, "8c")
    else:
      local_hist = local_mixed_hist(box)
    print ("local_hist = {0}".format(local_hist))
    # List concatenation
    img_hist = img_hist + local_hist
  img_hist = normalize(img_hist)
  print ("normalized hist = {0}".format(img_hist))
  return img_hist

def local_mixed_hist(box):
  hist_4c = local_cc_hist(box, "4c")
  hist_8c = local_cc_hist(box, "8c")
  return hist_4c + hist_8c

def local_cc_hist(box, method_name):
  hist = [0] * 16
  # for each black point do this
  height, width = box.shape
  for i in range(height):
    for j in range(width):
      if box[i][j] == False:
        # This is a black point
        print ("black point [{0}, {1}]".format(i, j))
        point = [i, j]
        bin_number = 0
        if method_name == "4c":
          directions = ["n", "s", "w", "e"]
        else:
          directions = ["nw", "sw", "ne", "se"]
        for k in range(len(directions)):
          if hit_cc(box, point, directions[k]):
            bin_number += 2 ** k
        hist[bin_number] += 1
  return hist

def hit_cc(box, starting_point, direction):
  if direction in ["n", "s", "w", "e"]:
    return hit_4c(box, starting_point, direction)
  else:
    return hit_8c(box, starting_point, direction)

def hit_4c(box, starting_point, direction):
  current_point = starting_point
  if direction == "n":
    return True in box[0 : starting_point[0], starting_point[1]]
  elif direction == "s":
    return True in box[starting_point[0]:, starting_point[1]]
  elif direction == "w":
    return True in box[starting_point[0], 0 : starting_point[1]]
  else:
    return True in box[starting_point[0], starting_point[1]:]

def hit_8c(box, starting_point, direction):
  if direction == "nw":
    while in_limits(box, current_point):
      current_point[0] -= 1
      current_point[1] -= 1
      if not in_limits(box, current_point):
        return False
      row = current_point[0]
      col = current_point[1]
      # If we hit something not background
      if box[row, col] == True:
        return True
  elif direction == "sw":
    while in_limits(box, current_point):
      current_point[0] += 1
      current_point[1] -= 1
      if not in_limits(box, current_point):
        return False
      row = current_point[0]
      col = current_point[1]
      if box[row, col] == True:
        return True
  elif direction == "ne":
    while in_limits(box, current_point):
      current_point[0] -= 1
      current_point[1] += 1
      if not in_limits(box, current_point):
        return False
      row = current_point[0]
      col = current_point[1]
      if box[row, col] == True:
        return True
  else:
    while in_limits(box, current_point):
      current_point[0] += 1
      current_point[1] += 1
      if not in_limits(box, current_point):
        return False
      row = current_point[0]
      col = current_point[1]
      if box[row, col] == True:
        return True

def normalize(hist):
  total = sum(hist)
  if total == 0:
    return 0
  for i in range(len(hist)):
    hist[i] = hist[i] / total
  return hist

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
