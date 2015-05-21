__author__ = 'lucas'

import numpy as np

class Vlad:


    def __init__(self, clusters_centers, descriptors_dimension):
        self.clusters_centers = clusters_centers
        self.n_clusters = len(clusters_centers)
        self.descriptors_dimension = descriptors_dimension

    def get_image_vlad(self, descriptors):
        dimension = descriptors.shape[1]
        if dimension != self.descriptors_dimension:
            raise ValueError("La dimension de los descriptores no calza con la dimension inicializada")
        descriptors_expanded = np.tile(descriptors.T,(self.n_clusters,1,1))
        n_descriptors = descriptors.shape[0]
        clusters_expanded = np.tile(self.clusters_centers.T,(n_descriptors,1,1))
        clusters_expanded = clusters_expanded.swapaxes(0,2)
        difference_matrix = descriptors_expanded - clusters_expanded
        distances_vector_clusters = np.sum(difference_matrix**2, axis=1)

        binary_nn_matrix = None
        for i in range(n_descriptors):
            column = np.zeros(self.n_clusters)
            column[np.argmax(distances_vector_clusters[:,i])] = 1
            if binary_nn_matrix is None:
                binary_nn_matrix = column
        else:
            binary_nn_matrix = np.column_stack((binary_nn_matrix,column))


        binary_nn_matrix_expaned = np.tile(binary_nn_matrix,(self.descriptors_dimension,1,1)).swapaxes(0,1)
        final_cube = binary_nn_matrix*difference_matrix
        return np.sum(final_cube, axis=2)









if __name__ == "__main__":
    print("hello world!")