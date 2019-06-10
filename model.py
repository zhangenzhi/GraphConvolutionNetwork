import tensorflow as tf
from layer import GraphConvLayer2
import matplotlib.pyplot as plt
from utilz import get_macro_data
import numpy as np


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
        supports.append(np.reshape(support,[12,1]))
    supports = np.asarray(supports)

    labels = data_dict["accuracy"]
    labels = np.reshape(labels,(len(labels),1))

    indice = [i for i in range(len(supports))]
    np.random.shuffle(indice)

    shuffled_supports = []
    shuffled_labels = []
    for i in indice:
        shuffled_supports.append(supports[i])
        shuffled_labels.append(labels[i])

    shuffled_supports = np.asarray(shuffled_supports)
    shuffled_labels = np.asarray(shuffled_labels)
    # need to make feature as data
    # need to compute D^(-0.5) and A+I
    model = tf.keras.Sequential()

    model.add(GraphConvLayer2(1,8))
    model.add(GraphConvLayer2(8,16))
    model.add(layers.Flatten())
    model.add(layers.Dense(768,activation="relu"))
    model.add(layers.Dense(16,activation="relu"))
    model.add(layers.Dense(1,activation="sigmoid"))


    model.compile(optimizer=tf.train.AdamOptimizer(0.01),
                loss='mse',metrics=['mae'])


    # val_data = np.random.random((100, 32))
    # val_labels = np.random.random((100, 10))

    history = model.fit(shuffled_supports[:3000], shuffled_labels[:3000], 
            epochs=100, batch_size=64,
            validation_split=0.2)

    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()