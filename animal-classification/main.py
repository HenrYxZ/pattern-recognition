import utils
import descriptors
import glob
import cv2
import time
import numpy as np
from dataset import Dataset
import cPickle as pickle

start = time.time()
# Define the training and testing sets
# path = "dataset"
# dataset = Dataset(path)
# dataset.generate_sets()
# pickle.dump(dataset, open("dataset.obj", "wb"), protocol=2)
dataset = pickle.load(open("dataset.obj", "rb"))

# Get SIFT descriptors per class
classes = dataset.get_classes()
train_set = dataset.get_train_set()
# Only using the first class
class_des = None
step = (len(train_set[0]) * 5) / 100
for i in range(len(train_set[0])):
	if i % step == 0:
		percentage = (i * 100) / len(train_set[0])
		print("Getting SIFT from image {0} of {1}({2}%) ...".format(
				i, len(train_set[0]), percentage
			)
		)
	img = cv2.imread(path)
	kp, des = descriptors.sift(img)
	if class_des == None:
		class_des = np.array(des, dtype=np.uint16)
	else:
		class_des = np.vstack((class_des, np.array(des, dtype=np.uint16)))
print("descriptors of shape = {0}".format(class_des.shape))
filename = "train/des_" + classes[0] + ".obj"
pickle.dump(class_des, open(filename, "wb"), protocol=2)
end = time.time()
print("Elapsed time processing {0}".format(utils.humanize_time(end - start)))