from tensorflow.keras.layers import Conv2D, LeakyReLU, MaxPool2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

from .reshape_yolo import YoloReshape

lrelu = LeakyReLU(alpha=0.1)
l2reg = l2(5e-4)


class YOLO:

  def __init__(self):
    # Graph
    self.layers = []
    self.model = None

    # Model
    self.build_layers()

  def build_layers(self):
    self.layers = []

    # Layer 1
    self.layers.append(Conv2D(filters=64,
                              kernel_size=(7, 7),
                              strides=(2, 2),
                              padding='same',
                              activation=lrelu,
                              kernel_regularizer=l2reg))

    self.layers.append(MaxPool2D(pool_size=(2, 2),
                                 strides=(2, 2),
                                 padding='same'))

    # Layer 2
    self.layers.append(Conv2D(filters=192,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              padding='same',
                              activation=lrelu,
                              kernel_regularizer=l2reg))
    self.add_maxpool_layer()

    # Add Paired Repeated layers
    # ------------------------

    # Layer 3
    self.add_repeated_layers(f1=128, f2=256, k1=(1, 1), k2=(3, 3), ntimes=1)
    self.add_repeated_layers(f1=256, f2=512, k1=(1, 1), k2=(3, 3), ntimes=1)
    self.add_maxpool_layer()

    # Layer 4
    self.add_repeated_layers(f1=256, f2=512, k1=(1, 1), k2=(3, 3), ntimes=4)
    self.add_repeated_layers(f1=512, f2=1024, k1=(1, 1), k2=(3, 3), ntimes=1)
    self.add_maxpool_layer()

    # Layer 5, 6
    self.add_repeated_layers(f1=512, f2=1024, k1=(1, 1), k2=(3, 3), ntimes=2)
    for i in range(4):
      self.layers.append(Conv2D(filters=1024,
                                kernel_size=(3, 3),
                                strides=(1, 1) if i != 1 else (2, 2),
                                padding='same',
                                activation=lrelu,
                                kernel_regularizer=l2reg))

    # Layer 7, 8
    self.layers.append(Flatten())
    self.layers.append(Dense(512))
    self.layers.append(Dropout(rate=0.5))
    self.layers.append(Dense(1470))
    self.layers.append(YoloReshape(target=(7, 7, 30)))

    self.model = Sequential(self.layers)
    print('-' * 50)
    print(f'total layers: {len(self.layers)}')

  def add_repeated_layers(self, f1, f2, k1, k2, ntimes=1):
    for ntimes in range(ntimes):
      self.layers.append(Conv2D(filters=f1,
                                kernel_size=k1,
                                strides=(1, 1),
                                padding='same',
                                activation=lrelu,
                                kernel_regularizer=l2reg))

      self.layers.append(Conv2D(filters=f2,
                                kernel_size=k2,
                                strides=(1, 1),
                                padding='same',
                                activation=lrelu,
                                kernel_regularizer=l2reg))
    return

  def add_maxpool_layer(self):
    self.layers.append(MaxPool2D(pool_size=(2, 2),
                                 strides=(2, 2),
                                 padding='same'))
    return

  def get_model(self):
    return self.model
