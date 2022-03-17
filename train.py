import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from tensorflow.python.data.ops.dataset_ops import DatasetV2 as Dataset


class TrainNN:

    def __init__(self):
        self.dataset = {}

    @staticmethod
    def generator(split):

        for i in range(len(globals()[f'{split}_dataframe'])):
            with h5py.File(Path.cwd().joinpath('dataset.h5'), 'r') as hdf:
                coords_array = hdf[split]['input'][str(i)][()]
                inp_dict = {'input': coords_array}
                class_array = hdf[split]['output'][str(i)][()]
                out_dict = {'output': class_array}

                yield inp_dict, out_dict

    @staticmethod
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

    def prepare_dataset(self):

        out_signature = [{}, {}]
        out_signature[0].update({'input': tf.TensorSpec(shape=(33, 3), dtype='float32')})
        out_signature[1].update({'output': tf.TensorSpec(shape=(3,), dtype='int8')})

        self.dataset = {'train': Dataset.from_generator(lambda: self.generator('train'),
                                                        output_signature=tuple(out_signature)),
                        'val': Dataset.from_generator(lambda: self.generator('val'),
                                                      output_signature=tuple(out_signature))}

    def start_train(self, epochs, batch_size, save_weights=False):

        model = self.make_model()
        model.fit(self.dataset['train'].batch(batch_size), epochs=epochs,
                  validation_data=self.dataset['val'].batch(batch_size))
        if save_weights:
            model.save_weights(Path.cwd().joinpath('weights', 'weights.h5'))
