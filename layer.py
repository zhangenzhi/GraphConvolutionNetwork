import tensorflow as tf
from tensorflow.keras import layers
from utilz import get_macro_data
import numpy as np


class GraphConvLayer(layers.Layer):

    def __init__(self,output_dim,**kwargs):
        self.output_dim = output_dim
        super(GraphConvLayer, self).__init__(**kwargs)
    
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
        #support * kernel ,support = D^(-0.5) * A' * D^(-0.5) * feature
        
        result = tf.matmul(inputs, self.kernel)
        return tf.nn.relu(result)
    
    def compute_output_shape(self,input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)
    
    def get_config(self):
        base_config = super(GraphConvLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
def NN():

    model = tf.keras.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(64,activation="relu"))
    model.add(layers.Dense(32,activation="relu"))
    model.add(layers.Dense(16,activation="relu"))
    model.add(layers.Dense(1,activation="sigmoid"))

    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                loss='mean_squared_error',metrics=['mae'])

    data_dict = get_macro_data()
    
    data = data_dict["A"] + data_dict["P"]
    data = np.asarray(data)

    labels = data_dict["accuracy"]
    labels = np.asarray(labels)

    # val_data = np.random.random((100, 32))
    # val_labels = np.random.random((100, 10))

    model.fit(data, labels, epochs=1000, batch_size=32)

def GCN():

    data_dict = get_macro_data()
    
    A = data_dict["A"] + np.transpose(data_dict["A"],axes=[0,2,1])
    D = data_dict["D"]
    feature = data_dict["P"]
    features = np.sum(feature,axis=2)


    data = np.asarray(A)
    datashape = np.shape(data)

    
    A_bar = A + np.eye(datashape[1],datashape[2])
    D_bar = D + np.eye(datashape[1],datashape[2])

    for i in range(len(D_bar)):
        for j in range(len(D_bar[0])):
            D_bar[i][j][j] = 1/np.sqrt(D_bar[i][j][j])
    
    supports = []
    for i in range(len(A_bar)):
        support = np.matmul(np.matmul(D_bar[i],A_bar[i]),D_bar[i])
        support = np.matmul(support,features[i])
        supports.append(support)
    supports = np.asarray(supports)

    labels = data_dict["accuracy"]
    labels = np.asarray(labels)

    # need to make feature as data
    # need to compute D^(-0.5) and A+I
    model = tf.keras.Sequential()

    model.add(GraphConvLayer(8))
    model.add(GraphConvLayer(16))
    model.add(layers.Flatten())
    model.add(layers.Dense(64,activation="relu"))
    model.add(layers.Dense(32,activation="relu"))
    model.add(layers.Dense(16,activation="relu"))
    model.add(layers.Dense(1,activation="sigmoid"))


    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                loss='mse',metrics=['mae'])


    # val_data = np.random.random((100, 32))
    # val_labels = np.random.random((100, 10))

    model.fit(supports, labels, epochs=1000, batch_size=32)



if __name__ == "__main__":
    GCN()