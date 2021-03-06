import numpy as np
import features
import glob
from scipy import misc
import re
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import random
import math
import time
import kmeans
import sys

###HOG OPTIONS
bins = 8
side_pixels_per_cell = 16
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

def optimize_parameters(X,y):
  exponents_C = range(-10,10)
  exponents_gamma = range(-10,10)
  best_c_exp = None
  best_gamma_exp = None
  best_accuracy = - np.inf
  for exp_C in exponents_C:
    for exp_gamma in exponents_gamma:
      print("Probando con C = 2^"+str(exp_C) +" y gamma = 2^"+str(exp_gamma)+" ... ")
      c = math.pow(2.0,exp_C)
      gamma_ = math.pow(2.0,exp_gamma)
      clf = SVC(kernel='rbf',C=c,gamma=gamma_)
      scores = cross_validation.cross_val_score(clf, X, y, cv=5)
      accuracy = scores.mean()
      print("Accurracy = " + str(accuracy))
      if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_c_exp = exp_C
        best_gamma_exp = exp_gamma
  return best_c_exp, best_gamma_exp, best_accuracy




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
      feats = np.vstack((feats, img_feats))
    # For debugging
    # if i < 5:
    #   print "Img features for img [{0}] = {1}".format(i, img_feats)
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
    training_file_list = glob.glob("Images_MNIST/Train/*")
    training_file_list = random.sample(training_file_list, 20000)
    testing_file_list = glob.glob("Images_MNIST/Test/*")
  else :
    training_file_list = glob.glob("Images_CVL/train/*")
    # training_file_list = random.sample(training_file_list, 1000)
    testing_file_list = glob.glob("Images_CVL/test/*")
    # testing_file_list = random.sample(testing_file_list, 250)

  ##############################################################################
  ###########                      Training                          ###########
  ##############################################################################
  print ("Starting to train using {0} images".format(len(training_file_list)))
  start = time.time()
  training_feats, training_classes = get_dataset(feats_option, dataset_option, training_file_list)
  end = time.time()
  print("Elapsed time getting training features: {0} secs".format(end - start))
  print(training_classes)

  X = training_feats
  y = training_classes

  ### OPTIMIZAR PARAMETROS
  #-----------------------------------------------------------------------------
  # start = time.time()
  # c, gamma, accuracy = optimize_parameters(X,y)
  # end = time.time()
  # print("Elapsed time optimizing: {0} secs".format(end - start))
  # print("best c exp = "),
  # print(c)
  # print("best gamma expc = "),
  # print(gamma)
  # print("best accuracy = "),
  # print(accuracy)
  # sys.exit()
  #-----------------------------------------------------------------------------

  clf = SVC(kernel='rbf', C=2**5, gamma=2**(-5))
  start = time.time()
  
  ### USANDO SVM
  #-----------------------------------------------------------------------------

  # clf.fit(X, y)
  #-----------------------------------------------------------------------------

  ### USANDO K MEANS
  #-----------------------------------------------------------------------------
  
  means = kmeans.get_means(X, y)
  print ("means = \n {0}".format(means))
  np.savetxt("means.csv", means, fmt = "%.6f", delimiter = ",")
  #-----------------------------------------------------------------------------

  end = time.time()
  print("Elapsed time training the classifier: {0} secs".format(end - start))


  ##############################################################################
  ###########                      Testing                           ###########
  ##############################################################################
  print ("Starting to test using {0} images".format(len(testing_file_list)))
  start = time.time()
  testing_feats, testing_classes = get_dataset(feats_option, dataset_option, testing_file_list)
  end = time.time()
  print("Elapsed time getting testing's features: {0} secs".format(end - start))
  
  start = time.time()
  X_test = testing_feats
  y_test_true = testing_classes

  ### USANDO SVM
  #-----------------------------------------------------------------------------

  # y_test_predicted = clf.predict(X_test)
  #-----------------------------------------------------------------------------

  ### USANDO K MEANS
  #-----------------------------------------------------------------------------
  
  y_test_predicted = kmeans.classify(X_test, means)
  #-----------------------------------------------------------------------------

  end = time.time()
  print("Elapsed time testing: {0} secs".format(end - start))
  
  print("testing_classes")
  print(testing_classes)
  
  confusion = confusion_matrix(y_test_true, y_test_predicted)
  print("confusion matrix = \n {0}".format(confusion))

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