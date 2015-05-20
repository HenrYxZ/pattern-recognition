class Model:
	""" The Model contains the important information obtained in the training.

	Attributes:
		clusters (array of numpy arrays of floats): The k clusters means.
		vlads (array of numpy arrays of floats): The VLAD descriptor of every
			image in the dataset.
	"""
	def __init__(self, clusters, vlads):
		self.clusters = clusters
		self.vlads = vlads