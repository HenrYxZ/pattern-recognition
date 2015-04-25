import numpy as np

def get_means(X, classes):
  D = len(X[0])
  n = len(X)
  k = 10
  cluster_sums = np.zeros((k, D))
  cluster_counts = [0] * 10
  means = np.zeros((k, D))
  # Get the sum of points for each class
  for i in range(n):
    this_class = classes[i]
    cluster_sums[this_class] += X[i]
    cluster_counts[this_class] += 1
  # Get the mean for each class by interpolation of points
  for j in range(k):
    if cluster_counts[j] > 0:
      means[j] = cluster_sums[j] / float(cluster_counts[j])
  return means

def assign(point, means, nearest_j):
  k = len(means)
  min_dist = np.linalg.norm(means[nearest_j] - point)
  # For each cluster find the distance from the point to the mean
  for j in range(k):
    if nearest_j != j:
      dist = np.linalg.norm(means[j] - point)
      if dist < min_dist:
        min_dist = dist
        nearest_j = j
  return nearest_j

def classify(X, means):
  Y = [assign(X[i], means, 0) for i in range(len(X))]
  return Y