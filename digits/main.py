import numpy as np
import features
import glob
from scipy import misc
import re
from sklearn.svm import SVC

###HOG OPTIONS
bins = 4
side_pixels_per_cell = 2
side_cells_per_block = 1

def get_class_MNIST(file_path):
    class_str = re.split('\-|\_|\.',file_path)[-2]
    return int(class_str)

def get_class_CVL(file_path):
    class_str = file_path.split('/')[-1].split('-')[0]
    return int(class_str)


#0 MNIST
#1 CVL

#0 train
#1 test
def get_file_list(dataset_option, train_or_test):
  if dataset_option == 0 and train_or_test == 0:
    return glob.glob('Images_MNIST/Train/*')
  elif dataset_option == 0 and train_or_test == 1:
    return glob.glob('Images_MNIST/Test/*')
  elif dataset_option == 1 and train_or_test == 0:
    return glob.glob('Images_CVL/train/*')
  else:
    return glob.glob('Images_CVL/test/*')


def get_dataset(feats_option, dataset_option, train_or_test):
  feats = None
  classes = []
  for file_path in file_list:
    image = misc.imread(file_path)
    ##  get features
    if feats_option == 1:
      img_feats = features.get_4c_feats(image)
    elif feats_option == 2:
      img_feats = features.get_8c_feats(image)
    elif feats_option == 3:
      img_feats = features.get_mixed_feats(image)
    else:
      img_feats = features.get_hog_feats(image,bins,side_pixels_per_cell,side_cells_per_block)
    if feats is None:
      feats = img_feats
    else:
      training_feats = np.vstack((feats,img_feats))

      classes.append(get_class_from_file_path(file_path))

    return (feats, np.array(classes))

################################################################################
#             MAIN
#-------------------------------------------------------------------------------
def main():
  # Read training images


  feats_option = input()


  print ("Choose an option for features: \n"\
         "[1] 4c\n"\
         "[2] 8c\n"\
         "[3] 4c concatenated with 8c\n"\
         "[4] HOG"
  )

  training_file_list = glob.glob("Images_CVL/train/*")[0:9]
  testing_file_list = glob.glob("Images_CVL/test/*")
  # training_file_list = glob.glob("Images_MNIST/Train/*")
  # testing_file_list = glob.glob("Images_MNIST/Test/*")


  ##############################################################################
  ###########                      Training                          ###########
  ##############################################################################

  training_feats, training_classes = get_dataset(training_file_list)


  ##############################################################################
  ###########                      Testing                           ###########
  ##############################################################################

  testing_feats, testing_classes = get_dataset(training_file_list)

if __name__ == '__main__':
  main()