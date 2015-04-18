class Classifier:
  
  """Classifier that uses a model previously generated in the training phase to
    classify input images.
  Attributes:
    model (matrix?): The model generated by the training.
  """

  def __init__(self, model):
    self.model = model

  # Returns a list with the class numbers assigned to each image in the image
  # features vector
  def evaluate(self, img_feats):
    pass
    