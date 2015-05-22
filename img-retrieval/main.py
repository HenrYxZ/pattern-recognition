import glob
import train
import test
import utils
import numpy as np
from time import time
from model import Model

def main():
	# img_files = ['oxbuild-images/all_souls_000000.jpg', ...]
	img_files = glob.glob("oxbuild-images/*.jpg")
	# img_names = ['all_souls_000000', ...]
	img_names = utils.get_images_names(img_files)
	# query_indices = [11, 21, ...]
	query_indices = utils.get_queries(img_names)
	files_for_codebook = img_files
	for index in query_indices:
		del files_for_codebook[index]

	# Training
	#---------------------------------------------------------------------------
	# Extracting descriptors
	
	start = time()
	descriptors_count = train.calculate_descriptors(files_for_codebook)
	end = time()
	elapsed_time = utils.humanize_time(end - start)
	print("Elapsed time getting the descriptors {0}.".format(elapsed_time))
	# Clustering
	
	# k = 64
	# start = time()
	# clusters = train.get_clusters(k)
	# end = time()
	# elapsed_time = utils.humanize_time(end - start)
	# print("Elapsed time clustering for k={0} {1}".format(k, elapsed_time))
	# np.savetxt("clusters64.csv", clusters, delimiter=",")
	
	# k = 128
	# start = time()
	# clusters = train.get_clusters(k)
	# end = time()
	# elapsed_time = utils.humanize_time(end - start)
	# print("Elapsed time clustering for k={0} {1}".format(k, elapsed_time))
	# np.savetxt("clusters128.csv", clusters, delimiter=",")
	
	# k = 256
	# start = time()
	# clusters = train.get_clusters(k)
	# end = time()
	# elapsed_time = utils.humanize_time(end - start)
	# print("Elapsed time clustering for k={0} {1}".format(k, elapsed_time))
	# np.savetxt("clusters256.csv", clusters, delimiter=",")

	
	# Testing
	#---------------------------------------------------------------------------

	# Hacer queries

if __name__ == '__main__':
	main()