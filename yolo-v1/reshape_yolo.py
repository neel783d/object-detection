import keras.backend as K
from tensorflow.keras.layers import Layer


class YoloReshape(Layer):
  def __init__(self, target_shape):
    super(YoloReshape, self).__init__()
    self.target_shape = tuple(target_shape)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
      'target_shape': self.target_shape
    })
    return config

  def call(self, input, **kwargs):
    # grids 7x7
    S = [self.target_shape[0], self.target_shape[1]]
    # classes
    C = 20
    # no of bounding boxes per grid
    B = 2

    idx1 = S[0] * S[1] * C
    idx2 = idx1 + S[0] * S[1] * B

    # multi-class classification
    class_probs = K.reshape(input[:, :idx1],
                            (K.shape(input)[0],) + tuple([S[0], S[1], C]))
    class_probs = K.softmax(class_probs)

    # confidence
    confs = K.reshape(input[:, idx1:idx2],
                      (K.shape(input)[0],) + tuple([S[0], S[1], B]))
    confs = K.sigmoid(confs)

    # boxes
    boxes = K.reshape(input[:, idx2:],
                      (K.shape(input)[0],) + tuple([S[0], S[1], B * 4]))
    boxes = K.sigmoid(boxes)

    outputs = K.concatenate([class_probs, confs, boxes])
    return outputs
