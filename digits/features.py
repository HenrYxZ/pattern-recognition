# Returns a list with the histogram concatenation
def get_cc_feats(image, cc_name):
  height, width, channels = image.shape
  # Use a threshold to get a binary image
  
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
      boxes.append(image[vertical_a : vertical_b, horizontal_a : horizontal_b])

  # Get the histogram for each box and concatenate them
  img_hist = []
  for box in range(len(boxes)):
    if (cc_name == "4c"):
      local_hist = local_4c_hist(box)
    elif (cc_name == "8c"):
      local_hist = local_8c_hist(box)
    else:
      local_hist = local_mixed_hist(box)
    # List concatenation
    img_hist += local_hist
  return img_hist

def local_4c_hist(box):
  for row in box

def local_8c_hist(box):
  pass

def local_mixed_hist(box):
  pass

def get_hog_feats(image):
  pass