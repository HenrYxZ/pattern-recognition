import numpy as np
import descriptors
import cv2
import glob
import utils
import scipy.io as sio
from scipy import spatial
import time

def all_classes_descriptors(dataset, opt = "sample"):
	des_files = glob.glob("train/*.mat")
	# train_set = dataset.get_train_set()
	
	# Read training descriptors
	print("Getting sample of the descriptors for classes")
	classes_des = []
	for c in range(n_classes):
		# class_files = train_set[c]
		path = des_files[c]
		print("Reading class number {0}".format(c))
		# class_des = class_descriptors(class_files)
		class_des = class_des_from_file(path)
		sample_size = 5000
		if opt == "sample":
			current_sample = utils.random_sample(class_des, sample_size)
			current_sample = np.array(current_sample, dtype=np.uint16)
			print("current sample shape = {0}".format(current_sample.shape))
			# class_des = None
			classes_des.append(current_sample)
		else:
			classes_des.append(class_des)

	end = time.time()
	elapsed_time = utils.humanize_time(end - start)
	print("Elapsed time getting training descriptors {0}".format(elapsed_time))
	return classes_des

def knn(dataset, des_classes):
	start = time.time()
	n_classes = len(dataset.get_classes())
	classes_sample = all_classes_descriptors(dataset)

	# Read testing descriptors
	start = time.time()
	test_folders = glob.glob("test/*")
	predictions = []
	counter = 0
	for folder in test_folders:
		print("Starting to predict the test set of class {0}.".format(counter))
		predictions.append([])
		# The i-th element of this list has the descriptors for the i-th image;
		# all the images are in the same class-folder
		test_files = glob.glob(folder + "/*.mat")
		# for i in range(len(test_files)):
		for i in range(25):
			distances = np.zeros(n_classes)
			# percentage = (i * 100) / len(test_files)
			percentage = (i * 100) / 25
			print(
				"Loading SIFT from file {0} of {1} ({2}%) class={3} ...".format(
				#	i, len(test_files), percentage, counter
					i, 25, percentage, counter
				)
			)
			fname = test_files[i]
			data = sio.loadmat(fname)
			des = data["stored"]
			# Find the nearest class 
			for c in range(n_classes):
				s = "Dist for img number {0} to class {1} (real = {2})".format(
					i, c, counter
				)
				print(s)
				class_des = classes_sample[c]
				distances[c] = dist_nn_class(des, class_des)
			predictions[-1].append(np.argmin(distances))
		counter += 1
	end = time.time()
	elapsed_time = utils.humanize_time(end - start)
	print("Elapsed time testing with KNN {0}".format(elapsed_time))
	return predictions

def knn_kdtree(dataset):
	# n_classes = len(dataset.get_classes())
	n_classes = 5
	des_files = glob.glob("train/*.mat")
	predictions = []
	counter = 0

	# Get KDTrees with the descriptors of the training set
	print("Getting sample of the descriptors for classes in their KD-trees")
	classes_trees = []
	start = time.time()
	for c in range(n_classes):
		fname = des_files[c]
		print("fname = {0}".format(fname))
		data = sio.loadmat(fname)
		class_des = data["stored"]

		sample_size = 1000
		current_sample = utils.random_sample(class_des, sample_size)
		tree = spatial.KDTree(current_sample)
		classes_trees.append(tree)
		# cleaning this variable
		class_des = None
	end = time.time()
	elapsed_time = utils.humanize_time(end - start)
	print("Elapsed time building KDTrees {0}".format(elapsed_time))

	for test_files in dataset.get_test_set():
		# Using only the first n_files objects of the test set for each class
		n_files = 15
		# TESTING! #
		if counter == n_classes:
			break
		############
		print("Starting to predict the test set of class {0}.".format(counter))
		# iteration over classes
		predictions.append([])
		# The i-th element of this list has the descriptors for the i-th image;
		# all the images are in the same class-folder
		test_des_list = []
		# step = (len(test_files) * 5) / 100
		# for i in range(len(test_files)):
		for i in range(n_files):
			# iteration over files inside a class
			# if i % step == 0:
			# 	percentage = (i * 100) / len(test_files)
			# 	print(
			# 		"Getting SIFT from file {0} of {1} ({2}%) ...".format(
			# 			i, len(test_files), percentage
			# 		)
			# 	)
			fname = test_files[i]
			test_img = cv2.imread(fname)
			kp, current_des = descriptors.sift(test_img)
			test_des_list.append(current_des)
		# We can't add more descriptors because it would be too expensive in RAM

		# distances = np.zeros((len(test_files), n_classes))
		distances = np.zeros((n_files, n_classes))
		for c in range(n_classes):
			# for img_index in range(len(test_files)):
			for img_index in range(n_files):
				des = test_des_list[img_index]
				print("Getting dist for img index = {0}".format(img_index))
				start = time.time()
				current_dists, ids = classes_trees[c].query(des)
				distances[img_index][c] = sum(current_dists)
				end = time.time()
				s = "Seconds getting distances {0} for {1} descriptors".format(
						(end - start), len(des) 
					)
				print(s)
		# for img_index in range(len(test_files)):
		for img_index in range(n_files):
			predictions[-1].append(np.argmin(distances[img_index]))
		counter += 1
	return predictions

				

def dist_nn_class(des, class_des):
	''' Gets the distance to a class using a list of descriptors. This
	is done using the distances from the descriptors to their neareast
	descriptors neighbors inside the class.

	Args:
		des (list of floats array): Descriptors of the query object.
		class_des (list of floats array): Descriptors of the class C.

	Returns:
		float: Sum of distances from each descriptor to their neareast neighbors
			of the class C.
	'''

	dist = 0.0
	step = len(des) / 4
	for i in range(len(des)):
		if i % step == 0:
			percentage = (i * 100) / len(des)
			print("Dist for descriptor {0} of {1}({2}%) ...".format(
					i, len(des), percentage
				)
			)
		current_des = des[i]
		dist += utils.min_dist(current_des, class_des)
	return dist

def class_des_from_file(path):
	data = sio.loadmat(path)
	return data["stored"]

def class_descriptors(class_files):
	class_des = None
	step = (len(class_files) * 5) / 100
	for i in range(len(class_files)):
		if i % step == 0:
			percentage = (i * 100) / len(class_files)
			print("Getting SIFT from image {0} of {1}({2}%) ...".format(
					i, len(class_files), percentage
				)
			)
		path = class_files[i]
		img = cv2.imread(path)
		kp, des = descriptors.sift(img)
		if class_des is None:
			class_des = np.array(des, dtype=np.uint16)
		else:
			class_des = np.vstack(
				(class_des, np.array(des, dtype=np.uint16))
			)	
	return class_des

