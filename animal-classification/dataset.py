import glob
import utils

class Dataset:
	''' This class manages the information for this particular dataset.
	'''
	def __init__(self, path):
		self.path = path
		self.train_set = []
		self.test_set = []
		self.classes = []

	def generate_sets(self):
		dataset_classes = glob.glob(self.path + "/*")
		for folder in dataset_classes:
			if "/" in folder:
				class_name = folder.split("/")[-1]
			else:
				class_name = folder.split("\\")[-1]
			self.classes.append(class_name)
			class_files = glob.glob(folder + "/*.JPEG")
			test_size = len(class_files) / 3
			train, test = utils.random_split(class_files, test_size)
			self.train_set.append(train)
			self.test_set.append(test)

	def get_train_set(self):
		if len(self.train_set) == 0:
			self.generate_sets()
		return self.train_set

	def get_test_set(self):
		if len(self.test_set) == 0:
			self.generate_sets()
		return self.test_set

	def get_classes(self):
		if len(self.classes) == 0:
			self.generate_sets()
		return self.classes

	def store_listfile(self):
		''' Used for creating files in the format filelist used in Caffe for
		converting an imageset. (caffe/tools/convert_imageset.cpp)
		'''
		train_file = open("train_file.txt", "w")
		test_file = open("test_file.txt", "w")
		self.get_train_set()
		self.get_test_set()
		for class_id in range(len(self.classes)):
			current_train = self.train_set[class_id]
			for filename in current_train:
				path = filename.replace("\\", "/")
				idx = path.index("/")
				path = path[(idx + 1):]
				train_file.write("{0} {1}\n".format(path, class_id))
			current_test = self.test_set[class_id]
			for filename in current_test:
				path = filename.replace("\\", "/")
				idx = path.index("/")
				path = path[(idx + 1):]
				test_file.write("{0} {1}\n".format(path, class_id))
		train_file.close()
		test_file.close()



