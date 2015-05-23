__author__ = 'lucas'

import numpy as np

class Vlad:


    def __init__(self, clusters_centers, descriptors_dimension):
        self.clusters_centers = clusters_centers
        self.n_clusters = clusters_centers.shape[0]
        self.descriptors_dimension = clusters_centers.shape[1]

    def get_image_vlad(self, descriptors):
        dimension = descriptors.shape[1]
        if dimension != self.descriptors_dimension:
            raise ValueError("La dimension de los descriptores no calza con la dimension inicializada")
        #print("la dimension de los descriptores es ok")
        descriptors_expanded = np.tile(descriptors.T,(self.n_clusters,1,1))
        #print("descriptors expanded shape: "),
        #print(descriptors_expanded.shape)
        n_descriptors = descriptors.shape[0]
        clusters_expanded = np.tile(self.clusters_centers.T,(n_descriptors,1,1))
        clusters_expanded = clusters_expanded.swapaxes(0,2)
        #print("clusters expanded shape: "),
        #print(clusters_expanded.shape)
        difference_matrix = descriptors_expanded - clusters_expanded
        distances_vector_clusters = np.sum(difference_matrix**2, axis=1)
        #print("distances vectors clusters shape: "),
        #print(distances_vector_clusters.shape)

        binary_nn_matrix = None
        for i in range(n_descriptors):
            column = np.zeros(self.n_clusters)
            column[np.argmax(distances_vector_clusters[:,i])] = 1
            if binary_nn_matrix is None:
                binary_nn_matrix = column
            else:
                binary_nn_matrix = np.column_stack((binary_nn_matrix,column))
        #print("binary nn shape: "),
        #print(binary_nn_matrix.shape)
        binary_nn_matrix_expanded = np.tile(binary_nn_matrix,(self.descriptors_dimension,1,1)).swapaxes(0,1)
        final_cube = binary_nn_matrix_expanded*difference_matrix
        vlad_matrix =  np.sum(final_cube, axis=2)
        flat_vlad = vlad_matrix.reshape((self.descriptors_dimension*self.n_clusters,))
        return normalize(flat_vlad)



def normalize(array):
    norm = np.linalg.norm(array)
    return array/norm







if __name__ == "__main__":
    print("hello world!")