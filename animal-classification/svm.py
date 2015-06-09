__author__ = 'lucas'

from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score
import cv2
import numpy as np
from vlad import Vlad
import descriptors
import matplotlib.pyplot as plt
from dataset import Dataset



#returns (X,y) tuple where X is the feature matrix and y is the class vector
def process_dataset(codebook, set_paths):
    vlad = Vlad(codebook)
    X = None
    y = []
    i = 0
    for class_paths in set_paths:
        for img_path in class_paths:
            image_descriptors = descriptors.sift(cv2.imread(img_path))
            image_vlad = vlad.get_image_vlad(image_descriptors)
            if X == None:
                X = image_vlad
            else:
                X = np.vstack((X, image_vlad))
        y += len(class_paths)*[i]
        i += 1
    return X, y

def svm_train(codebook, training_set_paths):
    X, y = process_dataset(codebook, training_set_paths)
    svm = LinearSVC()
    svm.fit(X, y)
    return svm



def svm_test(svm, codebook, testing_set_paths):
    X, y_real = process_dataset(codebook, testing_set_paths)
    y_predicted = svm.predict(X)
    cm = confusion_matrix(y_real, y_predicted)
    accuracy = accuracy_score(y_real, y_predicted)
    print("Acuracy {0}".format(accuracy))
    plot_confusion_matrix(cm, "confusion matrix (accuracy {0})".format(accuracy))


def generate_codebook(k, sample):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(sample)
    return kmeans.cluster_centers_


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title)


def main():
    sample = descriptors.random_sample(100000)
    print("sample obtenido")
    # print(sample)
    # print(sample.shape)
    k = 64
    #clustering
    print("Haciendo clustering...")
    codebook = generate_codebook(k, sample)
    np.savetxt("codebook60",codebook,delimiter=',')
    dataset = pickle.load('dataset.obj', "rb")
    #train
    print("Entrenando...")
    svm = svm_train(codebook, dataset.get_train_set())
    #test
    print("Testeando...")
    svm_test(svm, codebook, dataset.get_test_set())

if __name__ == "__main__":
    main()


