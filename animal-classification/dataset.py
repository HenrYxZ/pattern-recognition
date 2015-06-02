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
