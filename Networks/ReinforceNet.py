from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop


def CreateReinfoceNetwork(observation_space, action_space, fc_layers):
    net = keras.Sequential()
    net.add(
        Dense(fc_layers[0], input_shape=observation_space.shape, kernel_initializer=keras.initializers.GlorotUniform(),
              activation="relu"))
    if len(fc_layers) > 1:
        for layer in fc_layers[1:]:
            net.add(Dense(layer, kernel_initializer=keras.initializers.GlorotUniform(), activation="relu"))
    net.add(Dense(action_space.n, kernel_initializer=keras.initializers.GlorotUniform(), activation="softmax"))
    return net

def CreateBaseNetwork(observation_space, fc_layers):
    net = keras.Sequential()
    net.add(
        Dense(fc_layers[0], input_shape=observation_space.shape, kernel_initializer=keras.initializers.GlorotUniform(),
              activation="relu"))
    if len(fc_layers) > 1:
        for layer in fc_layers[1:]:
            net.add(Dense(layer, kernel_initializer=keras.initializers.GlorotUniform(), activation="relu"))
    net.add(Dense(1, kernel_initializer=keras.initializers.GlorotUniform(), activation=None))
    return net
