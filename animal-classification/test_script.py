import descriptors
import cPickle as pickle
import dataset
import classifiers

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
	predictions = classifiers.knn(dataset)
	print("predictions 10x10 = {0}".format(predictions[:10][:10]))
	hist = np.zeros((10, 10), dtype=np.uint32)
	for i in range(len(predictions)):
		for j in range(len(predictions[i])):
			c = predictions[i][j]
			hist[i][c] += 1
	print("hist = {0}".format(hist))


if __name__ == '__main__':
	test_knn()