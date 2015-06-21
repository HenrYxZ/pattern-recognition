import utils
import descriptors
import glob
import cv2
import time
import numpy as np
from dataset import Dataset
import scipy.io as sio
import pickle

def main():

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
	for i in range(5, len(classes)):
		class_name = classes[i]
		class_files = train_set[i]
		print("Getting descriptors for class {0} of length {1}".format(
				class_name, len(class_files)
			)
		)
		store_train_des(class_files, class_name)
	end = time.time()
	s = "Elapsed time processing {0}".format(utils.humanize_time(end - start))
	print(s)


def store_train_des(filenames, class_name):
	class_des = None
	step = (len(filenames) * 5) / 100
	for i in range(len(filenames)):
		path = filenames[i]
		if i % step == 0:
			percentage = (i * 100) / len(filenames)
			print("Getting SIFT from image {0} of {1}({2}%) ...".format(
					i, len(filenames), percentage
				)
			)
		img = cv2.imread(path)
		kp, des = descriptors.sift(img)
		if class_des == None:
			class_des = np.array(des, dtype=np.uint16)
		else:
			class_des = np.vstack((class_des, np.array(des, dtype=np.uint16)))
	print("descriptors of shape = {0}".format(class_des.shape))
	filename = "train/des_" + class_name + ".mat"
	data = {"stored": class_des}
	sio.savemat(filename, data)

def store_test_des(dataset):
	''' Stores the descriptors for the testing set in one file per object.
		Each file has a matrix where a row represent a descriptor.
	'''
	# e.g. test = [[bear/img1.jpg, bear/img2.jpg], [cat/img1.jpg, ...], ...]
	test = dataset.get_test_set()
	counter = 0
	n_files = 30
	folders = glob.glob("test/*")
	for files in test:
		s = "Getting descriptors in class {0}.".format(counter)
		print(s)
		for i in range(n_files):
			# iteration over files inside a class
			# if i % step == 0:
			percentage = (i * 100) / n_files
			print(
				"Getting SIFT from file {0} of {1} ({2}%) ...".format(
					i, n_files, percentage
				)
			)
			fname = files[i]
			test_img = cv2.imread(fname)
			kp, current_des = descriptors.sift(test_img)
			
			# Store current descriptors
			if "/" in fname:
				obj_name = fname.split("/")[-1]
				obj_name = obj_name.split(".")[0]
				storage_name = folders[counter] + "/" + obj_name + ".mat"
			else:
				obj_name = fname.split("\\")[-1]
				obj_name = obj_name.split(".")[0]
				storage_name = folders[counter] + "\\" + obj_name + ".mat"
			data = {"stored": current_des}
			sio.savemat(storage_name, data)
		counter += 1

if __name__ == '__main__':
	main()