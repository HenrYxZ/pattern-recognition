import numpy as np

def get_means(X, k):
  D = len(X[0])
  n = len(X)
  means = np.random.random_sample(size = (k, D))
  changed = True
  assignation = [0] * n  

  while changed:
    changed = False
    cluster_changed = [False] * k

    ### Assignation of cluster for each point
    #---------------------------------------------------------------------------
    # For each instance object vector point find the nearest mean
    for i in range(n):
      nearest_j = assign(X[i], means, assignation[i])
      # If the point changed the cluster
      if nearest_j != assignation[i]:
        changed = True
        cluster_changed[assignation[i]] = True
        # Assign the point to the new cluster
        assignation[i] = nearest_j

    ### New mean for each cluster changed
    #---------------------------------------------------------------------------
    for j in range(k):
      if cluster_changed[j]:
        vector_sum = np.zeros(D)
        count = 0
        for i in range(n):
          # If this point was assigned to this cluster
          if assignation[i] == j:
            vector_sum = vector_sum + X[i]
            count += 1
        # Recalculate mean
        if count > 0:
          means[j] = vector_sum / count

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