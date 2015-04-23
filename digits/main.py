import numpy as np
import features
import glob
from scipy import misc
import re
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import time

###HOG OPTIONS
bins = 8
side_pixels_per_cell = 7
side_cells_per_block = 3


def get_class_from_file_path(file_path, dataset_option):
  if dataset_option == 1:
    return get_class_MNIST(file_path)
  else:
    return get_class_CVL(file_path)

def get_class_MNIST(file_path):
    class_str = re.split('\-|\_|\.',file_path)[-2]
    return int(class_str)

def get_class_CVL(file_path):
    class_str = file_path.split('/')[-1].split('-')[0]
    return int(class_str)



#1 MNIST
#2 CVL

#0 train
#1 test


def get_dataset(feats_option, dataset_option, file_list):
  feats = None
  classes = []
  for i in range(len(file_list)):
    file_path = file_list[i]
    percentage = i * 100 / len(file_list)
    step = (len(file_list) *  5) / 100
    if i % step == 0:
      print ("Done with {0}% of the images".format(percentage))
    image = misc.imread(file_path)
    ##  get features
    if feats_option == 1:
      img_feats = features.get_cc_feats(image, "4c")
    elif feats_option == 2:
      img_feats = features.get_cc_feats(image, "8c")
    elif feats_option == 3:
      img_feats = features.get_cc_feats(image, "mixed")
    else:
      img_feats = features.get_hog_feats(image,bins,side_pixels_per_cell,side_cells_per_block)
    if feats is None:
      feats = img_feats
    else:
      feats = np.vstack((feats,img_feats))
    # For debugging
    if i < 5:
      print "Img features for img [{0}] = {1}".format(i, img_feats)

    classes.append(get_class_from_file_path(file_path, dataset_option))
  return feats, np.array(classes)

################################################################################
#             MAIN
#-------------------------------------------------------------------------------
def main():
  # Read training images

  print ("Choose an option for features: \n"\
         "[1] 4c\n"\
         "[2] 8c\n"\
         "[3] 4c concatenated with 8c\n"\
         "[4] HOG"
  )

  feats_option = input()

  print ("Choose an option for dataset: \n"\
         "[1] MNIST\n"\
         "[2] CVL"\
  )

  dataset_option = input()

  if dataset_option == 1:
    training_file_list = glob.glob("Images_MNIST/Train/*")[0:5000]
    testing_file_list = glob.glob("Images_MNIST/Test/*")[0:500]
  else :
    training_file_list = glob.glob("Images_CVL/train/*")[0:200]
    testing_file_list = glob.glob("Images_CVL/test/*")[0:50]

  ##############################################################################
  ###########                      Training                          ###########
  ##############################################################################
  start = time.time()
  training_feats, training_classes = get_dataset(feats_option, dataset_option, training_file_list)
  end = time.time()
  print("Elapsed time getting training features: {0} secs".format(end - start))
  print(training_classes)

  # print("training feats")
  # print(training_feats.shape)
  # print(training_feats)
  # print("")
  # print("training classes")
  # print(training_classes.shape)
  # print(training_classes)
  X = training_feats
  y = training_classes

  clf = SVC(kernel='linear')
  start = time.time()
  clf.fit(X, y)
  end = time.time()
  print("Elapsed time training the classifier: {0} secs".format(end - start))


  ##############################################################################
  ###########                      Testing                           ###########
  ##############################################################################

  start = time.time()
  testing_feats, testing_classes = get_dataset(feats_option, dataset_option, testing_file_list)
  end = time.time()
  print("Elapsed time getting testing's features: {0} secs".format(end - start))

  start = time.time()
  X_test = testing_feats
  y_test_true = testing_classes
  y_test_predicted = clf.predict(X_test)
  end = time.time()
  print("Elapsed time testing: {0} secs".format(end - start))

  print("testing_classes")
  print(testing_classes)

  confusion = confusion_matrix(y_test_true, y_test_predicted)
  accuracy = accuracy_score(y_test_true, y_test_predicted)
  print("Accuracy = "),
  print(accuracy)

    # Show confusion matrix
  plt.matshow(confusion)
  plt.title('Confusion matrix')
  plt.colorbar()
  plt.show()

if __name__ == '__main__':
  main()