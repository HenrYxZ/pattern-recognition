import numpy as np
import features
from classifier import Classifier

# Returns a data structure containing a model for classification
def train(feats_vec):
  model = []
  return model

# Returns an np matrix with the information of the confusion matrix
def get_confusion_matrix(results):
  global classes_count
  matrix = np.zeros((classes_count, classes_count))
  return matrix

# Returns the precision using the confusion matrix
def get_precision(confusion):
  return precision

################################################################################
#             MAIN
#-------------------------------------------------------------------------------
def main():
  # Read training images
  features_vector = []
  print ("Choose an option for features: \n"\
         "[1] 4c\n"\
         "[2] 8c\n"\
         "[3] 4c concatenated with 8c\n"\
         "[4] HOG"
  )
  feats_option = input()

  ##############################################################################
  ###########                      Training                          ###########
  ##############################################################################

  # for each image in training:
  ##  get features
  if feats_option == 1:
    img_feats = features.get_4c_feats(img)
  elif feats_option == 2:
    img_feats = features.get_8c_feats(img)
  elif feats_option == 3:
    img_feats = features.get_mixed_feats(img)
  else:
    img_feats = features.get_hog_feats(img)

  features_vector.append(img_feats)

  model = train(features_vector)
  classifier = Classifier(model)

  ##############################################################################
  ###########                      Testing                           ###########
  ##############################################################################
  
  testing_feats = []
  # for each image in testing
  if feats_option == 1:
    img_feats = features.get_4c_feats(img)
  elif feats_option == 2:
    img_feats = features.get_8c_feats(img)
  elif feats_option == 3:
    img_feats = features.get_mixed_feats(img)
  else:
    img_feats = features.get_hog_feats(img)
  
  testing_feats.append(img_feats)
  results = classifier.evaluate(testing_feats)

  confusion = 


if __name__ == '__main__':
  main()