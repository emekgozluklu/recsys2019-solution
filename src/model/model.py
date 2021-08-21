from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib


class NN:

    def __init__(self, train_path="../../data/normalized_features.csv", test_path=""):

        self.data = None
        self.epochs = None
        self.batch_size = None

        if train_path:
            self.data = pd.read_csv(train_path).astype(float)

        self.validation_split = self.data[self.data["is_validation"] == 1].copy()
        self.train_split = self.data[self.data["is_validation"] != 1].copy()

        self.train_Y = self.train_split["clicked"]
        self.train_X = self.train_split.drop(columns=["is_validation", "clicked"])

        self.validation_Y = self.validation_split["clicked"]
        self.validation_X = self.validation_split.drop(columns=["is_validation", "clicked"])

        self.validation_X.to_csv("valx.csv")
        self.validation_Y.to_csv("valy.csv")

        del self.data, self.train_split, self.validation_split

        assert self.train_X.shape[0] == self.train_Y.shape[0]
        assert self.validation_X.shape[0] == self.validation_Y.shape[0]

        self.input_dimension = self.train_X.shape[1]
        self.training_samples = self.train_X.shape[0]
        self.validation_samples = self.validation_X.shape[0]

        self.history = None
        self.model = None

    def build_model(self, dimensions=(120, 64)):
        self.model = Sequential()
        self.model.add(Dense(dimensions[0], input_dim=self.input_dimension, activation="relu"))
        self.model.add(Dense(dimensions[1], activation="relu"))
        self.model.add(Dense(1, activation="sigmoid"))

        return self.model

    def train(self, epochs=50, batch_size=128):
        self.epochs = epochs
        self.batch_size = batch_size

        opt = SGD(lr=0.01)

        self.model.compile(
            optimizer="rmsprop",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        self.history = self.model.fit(
            x=self.train_X, y=self.train_Y,
            validation_data=(self.validation_X, self.validation_Y),
            epochs=self.epochs, batch_size=self.batch_size
        )

        self.model.save_weights("../../data/model.h5")
        # joblib.dump(self.history, "../../data/hist.joblib")

    def plot(self):
        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, self.epochs), self.history.history["loss"], label="train_loss")
        plt.plot(np.arange(0, self.epochs), self.history.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, self.epochs), self.history.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, self.epochs), self.history.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.savefig("../../fig.png")


if __name__ == "__main__":
    nn_model = NN()
    nn_model.build_model()
    nn_model.train()
    nn_model.plot()

