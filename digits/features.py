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
  # Get the local histogram for each box and concatenate them
  for box in boxes:
    # print ("box = {0}".format(box))
    # print ("box shape = {0}".format(box.shape))
    if (cc_name == "4c"):
      local_hist = local_cc_hist(box, "4c")
    elif (cc_name == "8c"):
      local_hist = local_cc_hist(box, "8c")
    else:
      local_hist = local_mixed_hist(box)
    
    # List concatenation
    img_hist = img_hist + local_hist
  # print ("image hist = {0}".format(img_hist))
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
        # print ("black point [{0}, {1}]".format(i, j))
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
  # print ("local_hist = {0}".format(hist))
  return normalize(hist)

def hit_cc(box, starting_point, direction):
  if direction in ["n", "s", "w", "e"]:
    return hit_4c(box, starting_point, direction)
  else:
    return hit_8c(box, starting_point, direction)

def hit_4c(box, starting_point, direction):
  current_point = starting_point
  if direction == "n":
    block = box[0 : starting_point[0], starting_point[1]]
  elif direction == "s":
    block = box[starting_point[0]:, starting_point[1]]
  elif direction == "w":
    block = box[starting_point[0], 0 : starting_point[1]]
  else:
    block = box[starting_point[0], starting_point[1]:]

  hit = True in block
  # if direction == "e" and hit:
  #   print ("dir = {0}\n block = {1}".format(direction, block))
  #   sys.exit()
  return hit

def hit_8c(box, starting_point, direction):
  row = starting_point[0]
  col = starting_point[1]
  while True:
    if direction == "nw":
      row -= 1
      col -= 1
    elif direction == "sw":
      row += 1
      col -= 1
    elif direction == "ne":
      row -= 1
      col += 1
    else:
      row += 1
      col += 1
    if not in_limits(box, [row, col]):
      break
    elif box[row, col] == True:
      # if direction == "se":
      #   print ("dir = {0}\n hit in = [{1}, {2}]".format(direction, row, col))
      #   sys.exit()
      return True
  return False


def normalize(hist):
  total = sum(hist)
  if total == 0:
    return 0
  else:
    return [element / float(total) for element in hist]

def in_limits(box, point):
  in_limits_rows = point[0] >= 0 and point[0] < len(box)
  in_limits_cols = point[1] >= 0 and point[1] < len(box[0])
  if in_limits_rows and in_limits_cols:
    return True
  else:
    # print ("Out of limit in point = {0}".format(point))
    return False


def get_hog_feats(image, bins, pixels_per_cell_side, cells_per_block_side):
  image = color.rgb2gray(image)
  #image_resized = resize(image,(64,64))
  fd = hog(image, orientations=bins, pixels_per_cell=(pixels_per_cell_side, pixels_per_cell_side),
                        cells_per_block=(cells_per_block_side, cells_per_block_side))
  return fd
