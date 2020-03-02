from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Sequential
from keras.layers import Dense

class ClassficationModel:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.model = self._initialize_model()

    def _initialize_model(self):
        model = Sequential()
        model.add(MobileNetV2(include_top=False, weights='imagenet', pooling='avg'))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(2, activation="softmax"))

        return model

    def train(self, data_generator):
        self.model.fit_generator(
            data_generator,
            steps_per_epoch=2000,
            epochs=50
        )