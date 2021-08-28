import cv2 as cv
import numpy as np
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):

  def __init__(self, dataset, batch_size):
    self.dataset = dataset
    self.batch_size = batch_size
    self.len = int(len(dataset) * 1.0 / batch_size)

  def __len__(self):
    return self.len

  def __getitem__(self, idx):
    batched_data = self.dataset[idx * self.batch_size:
                                (idx + 1) * self.batch_size]
    train, labels = list(zip(*map(lambda data: self.read(data), batched_data)))
    return np.array(train), np.array(labels)

  @staticmethod
  def read(data):
    data = data.split('\t')
    path = data[2]

    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image_w, image_h = image.shape[0:-1]
    print(image.shape[0:-1])

    # reshaping image into 448, 448
    image = cv.resize(image, (448, 448))
    image = image / 255.

    # Ground truth: 7x7x30
    label_matrix = np.zeros([7, 7, 30])

    xmin = int(data[-4])
    xmax = int(data[-3])
    ymin = int(data[-2])
    ymax = int(data[-1])
    cls = int(data[1])

    # Scaled Co-ordinates
    x = (xmin + xmax) / 2 / image_w
    y = (ymin + ymax) / 2 / image_h
    w = (xmax - xmin) / image_w
    h = (ymax - ymin) / image_h

    # Location in the grid
    i, j = int(7 * x), int(7 * y)

    # Fractions on values
    x, y = x - i, y - j

    # Updating Label Matrix
    # label index --> 20, 21, 22, 23, 24 --> [x, y, w, h, response]
    if label_matrix[i, j, 24] == 0:
      # Bounding Box Exist: first 20 values
      label_matrix[i, j, cls] = 1

      # Next 4 values for bounding box shape
      label_matrix[i, j, 20:24] = [x, y, w, h]

      # Last value for respone
      label_matrix[i, j, 24] = 1

    return image, label_matrix
