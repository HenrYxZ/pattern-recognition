import numpy.random as nprnd
import numpy as np

def random_split(l, sample_size):
	sample_indices = nprnd.choice(len(l), size=sample_size, replace=False)
	# print (len(sample_indices))
	sample_indices.sort()
	# print("sample_indices = {0}".format(sample_indices))
	other_part = []
	sample_part = []
	indices_counter = 0
	for index in range(len(l)):
		current_elem = l[index]
		if indices_counter == sample_size:
			other_part = other_part + l[index:]
			break
		if index == sample_indices[indices_counter]:
			sample_part.append(current_elem)
			indices_counter += 1
		else:
			other_part.append(current_elem)
	return other_part, sample_part

def random_sample(l, sample_size):
	sample_indices = nprnd.choice(len(l), size=sample_size, replace=False)
	sample_indices.sort()
	return [l[index] for index in sample_indices]

def min_dist(query_point, points):
	min_dist = float("inf")
	for i in range(len(points)):
		p = points[i]
		diff_vec = query_point - p
		dist = np.dot(diff_vec, diff_vec)
		if dist < min_dist:
			min_dist = dist
	return min_dist

def humanize_time(secs):
	'''
	Extracted from http://testingreflections.com/node/6534
	'''
	mins, secs = divmod(secs, 60)
	hours, mins = divmod(mins, 60)
	return '%02d:%02d:%02d' % (hours, mins, secs)