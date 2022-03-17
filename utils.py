from tensorflow import keras
from tensorflow.keras import layers


def make_model():

    inputs = keras.Input(shape=(33, 3), name='input')
    x = layers.Conv1D(64, 5, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(128, 3, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(3, activation="softmax", name='output')(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

    return model
