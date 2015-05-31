import cv2
import glob
import descriptors
import utils

# Testing SIFT descriptors
#-------------------------------------------------------------------------------

# img_path = glob.glob("dataset/cat/*.JPEG")[0]
# img = cv2.imread(img_path)
# cv2.imshow("image", img)
# cv2.waitKey(0)
# des = descriptors.sift(img)
# print "descriptors len = {0}".format(len(des))
# print "descriptors[0] len = {0}".format(len(des[0]))

# Testing utils
#-------------------------------------------------------------------------------

l = range(40)
sample_size = 5
reminder, sample = utils.random_split(l, sample_size)
print ("reminder = {0}".format(reminder))
print ("sample = {0}".format(sample))