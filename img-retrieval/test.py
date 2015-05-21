import utils

def query(gray_img, region):
	x0, y0 = region[0]
	x1, y1 = region[1]
	data = gray_img[x0:x1, y0:y1]
	desc = utils.get_descriptors(data, region)
	vlad = get_vlad(desc, model.clusters)
	ranking = get_ranking(vlad, model)
	# Ver cuantos del ranking salieron correctos segun el groundtruth

def get_ranking(vlad, model):
	distances = utils.get_distances(vlad, model.vlads, "euclidean")
	tuples = []
	for i in range(len(distances)):
		tuples.append((i, distances))
	sorted_tuples = sorted(tuples, key=get_key)
	return [sorted_tuples[i][1] for i in range(len(distances))]

def get_groundtruth():
	pass
	

def get_key(x):
	return x[1]