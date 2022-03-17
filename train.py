import h5py
import tensorflow as tf
import sys
from utils import make_model
from pathlib import Path
from tensorflow.python.data.ops.dataset_ops import DatasetV2 as Dataset


class TrainNN(object):

    dataset = {}
    dataframe_length = {'train': 24020, 'val': 5988}

    def generator(self, split):

        for i in range(self.dataframe_length[split]):
            with h5py.File(Path.cwd().joinpath('dataset', 'dataset.h5'), 'r') as hdf:
                coords_array = hdf[split]['input'][str(i)][()]
                inp_dict = {'input': coords_array}
                class_array = hdf[split]['output'][str(i)][()]
                out_dict = {'output': class_array}

                yield inp_dict, out_dict

    def prepare_dataset(self):

        out_signature = [{}, {}]
        out_signature[0].update({'input': tf.TensorSpec(shape=(33, 3), dtype='float32')})
        out_signature[1].update({'output': tf.TensorSpec(shape=(3,), dtype='int8')})

        self.dataset = {'train': Dataset.from_generator(lambda: self.generator('train'),
                                                        output_signature=tuple(out_signature)),
                        'val': Dataset.from_generator(lambda: self.generator('val'),
                                                      output_signature=tuple(out_signature))}

    def start_train(self, epochs, batch_size):

        model = make_model()
        model.fit(self.dataset['train'].batch(batch_size), epochs=epochs,
                  validation_data=self.dataset['val'].batch(batch_size))
        model.save_weights(Path.cwd().joinpath('weights', 'weights.h5'))


train = TrainNN()
train.prepare_dataset()
train.start_train(int(sys.argv[1]), int(sys.argv[2]))
