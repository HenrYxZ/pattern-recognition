import glob
import train
import test
import utils
import numpy as np
from time import time
from model import Model

def main():
	img_files = glob.glob("oxbuild-images/*.jpg")
	
	# Training
	#---------------------------------------------------------------------------
	# Extracting descriptors
	start = time()
	descriptors = train.get_descriptors(img_files[0:100])
	end = time()
	elapsed_time = utils.humanize_time(end - start)
	print("Elapsed time getting the descriptors {0}.".format(elapsed_time))
	# Clustering
	k = 64
	start = time()
	clusters = train.get_clusters(k, descriptors)
	end = time()
	elapsed_time = utils.humanize_time(end - start)
	print("Elapsed time clustering for k={0} {1}".format(k, elapsed_time))
	np.savetxt("clusters64.csv", clusters, delimiter=",")
	
	# Testing
	#---------------------------------------------------------------------------

	# Hacer queries

if __name__ == '__main__':
	main()