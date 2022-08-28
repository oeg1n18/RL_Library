from tensorflow import keras
from tensorflow.keras.layers import Dense


def CreateQNetwork(observation_space, action_space, fc_layers):
    Qnet = keras.Sequential()
    Qnet.add(Dense(fc_layers[0], input_shape=observation_space.shape, activation="elu"))
    if len(fc_layers) > 1:
        for layer in fc_layers[1:]:
            Qnet.add(Dense(layer, activation="elu"))
    Qnet.add(Dense(action_space.n, activation="relu"))
    Qnet.compile(loss="mse", optimizer="adam")
    return Qnet
