import glob
import train
import test
import utils
import numpy as np
from time import time
from model import Model

def main():
	img_files = glob.glob("/home/lucas/Copy/Cursos PUC/patrones/imagenes tarea 3/images/*.jpg")
	
	# Training
	#---------------------------------------------------------------------------
	start = time()
	descriptors = train.get_descriptors(img_files[0:100])
	end = time()
	elapsed_time = utils.humanize_time(end - start)
	print("Elapsed time getting the descriptors {0}.".format(elapsed_time))
	k = 64
	clusters = train.get_clusters(k, descriptors)
	np.savetxt("clusters64.csv", clusters, delimiter=",")
	
	# Testing
	#---------------------------------------------------------------------------

	# Hacer queries

if __name__ == '__main__':
	main()