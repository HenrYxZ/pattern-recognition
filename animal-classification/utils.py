import numpy.random as nprnd

def random_split(l, sample_size):
	sample_indices = nprnd.choice(len(l), size=sample_size, replace=False)
	#print (len(sample_indices))
	sample_indices.sort()
	print("sample_indices = {0}".format(sample_indices))
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