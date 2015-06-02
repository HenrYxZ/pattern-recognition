import cv2
import glob
import descriptors
import utils
import numpy as np

# Testing SIFT descriptors
#-------------------------------------------------------------------------------
def test_sift():
	img_path = glob.glob("dataset/bear/*.JPEG")[0]
	img = cv2.imread(img_path)
	kp, des = descriptors.sift(img)
	img = cv2.drawKeypoints(img, kp, color=(255,0,0))
	cv2.imshow("image", img)
	cv2.waitKey(0)
	print("descriptors len = {0}".format(len(des)))
	print("descriptors[0] len = {0}".format(len(des[0])))
	print("descriptors[0][:5] = {0}".format(np.array(des[0][:], dtype=np.uint16)))

# Testing utils
#-------------------------------------------------------------------------------
def test_randsplit():
	l = range(40)
	sample_size = 5
	reminder, sample = utils.random_split(l, sample_size)
	print("reminder = {0}".format(reminder))
	print("sample = {0}".format(sample))

# Testing FAST keypoints
#-------------------------------------------------------------------------------
def test_fast():
	img_path = glob.glob("dataset/bear/*.JPEG")[0]
	img = cv2.imread(img_path)
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	fast = cv2.FastFeatureDetector(50)
	keypoints = fast.detect(img, None)
	img = cv2.drawKeypoints(img, keypoints, color=(255,0,0))
	cv2.imshow("image", img)
	cv2.waitKey(0)
	print("keypoints len = {0}".format(len(keypoints)))

################################################################################
####                               MAIN                                     ####
################################################################################
if __name__ == '__main__':
	test_sift()