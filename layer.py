import tensorflow as tf
from tensorflow.keras import layers
from utilz import get_macro_data
import numpy as np


class GraphConvLayer(layers.Layer):
    def __init__(self,output_dim,**kwargs):
        self.output_dim = output_dim
    
    def build(self,input_shape):
        shape = tf.TensorShape((input_shape[1], self.output_dim))
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                    shape=shape,
                                    initializer='uniform',
                                    trainable=True)

        # Be sure to call this at the end
        super(GraphConvLayer, self).build(input_shape)
    
    def call(self,inputs):
        return tf.matmul(inputs, self.kernel)
    
    def compute_output_shape(self,input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)
    
    def get_config(self):
        base_config = super(GraphConvLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

def NN():
    model = tf.keras.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(128,activation="relu"))
    model.add(layers.Dense(64,activation="relu"))
    model.add(layers.Dense(32,activation="relu"))
    model.add(layers.Dense(16,activation="relu"))
    model.add(layers.Dense(1,activation="sigmoid"))

    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                loss='mean_squared_error')

    data_dict = get_macro_data()
    
    data = data_dict["A"] + data_dict["P"]
    data = np.asarray(data)

    labels = data_dict["accuracy"]
    labels = np.asarray(labels)

    # val_data = np.random.random((100, 32))
    # val_labels = np.random.random((100, 10))

    model.fit(data, labels, epochs=1000, batch_size=32)




if __name__ == "__main__":
    test()