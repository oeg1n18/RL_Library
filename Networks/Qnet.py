from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop


def CreateQNetwork(observation_space, action_space, fc_layers):
    Qnet = keras.Sequential()
    Qnet.add(Dense(fc_layers[0], input_shape=observation_space.shape, kernel_initializer=keras.initializers.GlorotUniform(),  activation="elu"))
    if len(fc_layers) > 1:
        for layer in fc_layers[1:]:
            Qnet.add(Dense(layer, kernel_initializer=keras.initializers.GlorotUniform(), activation="elu"))
    Qnet.add(Dense(action_space.n, kernel_initializer=keras.initializers.GlorotUniform(), activation="linear"))
    Qnet.compile(loss="mse", optimizer=RMSprop())
    return Qnet
