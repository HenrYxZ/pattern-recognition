import descriptors
import cPickle as pickle
import dataset
import classifiers
import time
import utils
import numpy as np
import matplotlib.pyplot as plt

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
	# dataset = pickle.load(open("dataset.obj", "rb"))
	# start = time.time()
	# predictions = classifiers.knn(dataset)
	# end = time.time()
	# elapsed_time = utils.humanize_time(end - start)
	# print("Elapsed time using knn {0}...".format(elapsed_time))
	# print("predictions 5x10 = {0}".format(predictions[:][:10]))
	predictions = [
		[1, 1, 0, 2, 4, 3, 2, 0, 2, 4, 0, 3, 2, 1, 1],
		[1, 2, 4, 2, 1, 0, 4, 1, 3, 2, 2, 2, 1, 2, 1],
		[2, 3, 4, 2, 2, 0, 2, 0, 3, 3, 1, 2, 2, 2, 3],
 		[0, 1, 3, 3, 3, 3, 1, 3, 3, 3, 2, 2, 3, 0, 1],
 		[3, 0, 2, 1, 4, 2, 1, 0, 2, 4, 1, 1, 4, 2, 3]
 	]
	hist = np.zeros((5, 5), dtype=np.uint32)
	for i in range(len(predictions)):
		for j in range(len(predictions[i])):
			c = predictions[i][j]
			hist[i][c] += 1
	print("hist = {0}".format(hist))
	confusion_matrix = hist / 15.0
	print("conf mat = \n{0}".format(confusion_matrix))
	values = [confusion_matrix[i][i] for i in range(5)]
	precision = np.average(values)
	print("precision = {0}".format(precision))

	plt.matshow(confusion_matrix)
	plt.title('Confusion matrix')
	plt.colorbar()
	plt.show()


if __name__ == '__main__':
	test_knn()