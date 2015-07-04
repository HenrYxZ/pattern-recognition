# EXTERNAL MODULES
import cPickle as pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
import cv2
import scipy.io as sio

# LOCAL MODULES
import wtahash as wh
import main
import utils
import dataset
import descriptors
import knn

def test_sift():
	opt = input(
		"Choose an option:\n"\
		" [0] Write the image in this folder\n"\
		" [1] Show image\n"
	)
	n = input("Enter the image number to use.\n")
	WRITE = (opt == 0)
	dataset = pickle.load(open("dataset.obj", "rb"))
	img_path = dataset.get_train_set()[8][n]
	first_img = cv2.imread(img_path)
	kp, des = descriptors.sift(first_img)
	img = cv2.drawKeypoints(first_img, kp)
	if WRITE:
		cv2.imwrite("sift.jpg", img)
	else:
		cv2.imshow("Keypoints", img)
		cv2.waitKey()

# Testing counts
def test_counts():
	# counts = descriptors.counts()
	# print("counts = {0}".format(counts))
	counts = [
		1085495, 761595, 1035142, 1195164, 721518, 950140, 1125565, 1005910,
		845804, 820759, 9547092
	]

# Testing random sample
def test_sample():
	sample = descriptors.random_sample(counts)

# Testing knn
def test_knn():
	dataset = pickle.load(open("dataset.obj", "rb"))
	n_classes = len(dataset.get_classes())
	start = time.time()
	predictions = knn.knn(dataset)
	end = time.time()
	elapsed_time = utils.humanize_time(end - start)
	print("Elapsed time using knn {0}...".format(elapsed_time))
	print("predictions = \n{0}".format(predictions))
	utils.write_list(predictions, "results/knn-predictions.txt")
	# predictions = [
	# 	[1, 1, 0, 2, 4, 3, 2, 0, 2, 4, 0, 3, 2, 1, 1],
	# 	[1, 2, 4, 2, 1, 0, 4, 1, 3, 2, 2, 2, 1, 2, 1],
	# 	[2, 3, 4, 2, 2, 0, 2, 0, 3, 3, 1, 2, 2, 2, 3],
 	#	[0, 1, 3, 3, 3, 3, 1, 3, 3, 3, 2, 2, 3, 0, 1],
 	# 	[3, 0, 2, 1, 4, 2, 1, 0, 2, 4, 1, 1, 4, 2, 3]
 	# ]
	hist = np.zeros((n_classes, n_classes), dtype=np.uint16)
	for i in range(len(predictions)):
		for j in range(len(predictions[i])):
			c = predictions[i][j]
			hist[i][c] += 1
	print("hist = \n{0}".format(hist))
	np.savetxt("results/knn-hist.csv", hist, fmt="%i", delimiter=",")
	confusion_matrix = hist / 25.0
	print("conf mat = \n{0}".format(confusion_matrix))
	values = [confusion_matrix[i][i] for i in range(n_classes)]
	precision = np.average(values)
	print("precision = {0}".format(precision))

	plt.matshow(confusion_matrix)
	plt.title('Confusion matrix')
	plt.colorbar()
	plt.show()

def test_kdtree():
	values = np.array([[1.21,2.3], [3.63, 4.89], [60.21, 89.01], [0.21, 0.1]])
	tree = spatial.KDTree(values)
	print("tree data=\n{0}".format(tree.data))
	print("len tree data = {0}".format(len(tree.data)))
	test = np.array([[1.1, 1.1], [100.2, 90.3], [0.1, 0.03], [5.2, 5.2]])
	nn = tree.query(test)
	print("nn = \n{0}".format(nn))

def test_store_descriptors():
	start = time.time()
	dataset = pickle.load(open("dataset.obj", "rb"))
	main.store_test_des(dataset)
	end = time.time()
	s = "Elapsed time processing {0}".format(utils.humanize_time(end - start))
	print(s)

def test_dataset_listfile():
	dataset = pickle.load(open("dataset.obj", "rb"))
	dataset.store_listfile()

def test_conf_mat():
	hist = np.loadtxt("results/knn-hist.csv", delimiter=",")
	precs = hist / 25.0
	values = [precs[i][i] for i in range(len(precs))]
	accuracy = np.average(values)
	print("accuracy = {0}".format(accuracy))

	plt.matshow(hist)
	plt.title('Confusion matrix')
	plt.colorbar()
	plt.show()

def test_create_hash():
	n = 32
	w = 2
	k = 16
	train_file = sio.loadmat("train/des_bear.mat")
	train_data = train_file["stored"]
	print ("Starting to generate hash table ...")
	start = time.time()
	wta_hash = wh.WTAHash(train_data, n, k, w)
	end = time.time()
	elapsed_time = utils.humanize_time(end - start)
	print("Elapsed time on generation of hash table: {0}".format(elapsed_time))
	print("Starting to store the hash ...")
	start = time.time()
	pickle.dump(
		wta_hash, open("bear_hash.obj", "wb"), protocol=pickle.HIGHEST_PROTOCOL
	)
	end = time.time()
	print("Elapsed time storing the hash {0} seconds".format(end - start))

if __name__ == '__main__':
	test_create_hash()
