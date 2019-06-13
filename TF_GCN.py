import tensorflow as tf
from utilz import get_shuffled_data
import numpy as np
import matplotlib.pyplot as plt


features = tf.placeholder(dtype=tf.float32,shape=[None,12,1],name="features")
supports = tf.placeholder(dtype=tf.float32,shape=[None,12,12],name="supports")
labels = tf.placeholder(dtype=tf.float32,shape=[None,1],name="labels")

supports_1 = tf.matmul(supports,features)
weight_1 = tf.get_variable("weight_1",[1,16])
result_1 = tf.nn.relu(tf.matmul(tf.reshape(supports_1,[-1,1]),weight_1))

support_2 = tf.matmul(supports,tf.reshape(result_1,[-1,12,16]))
weight_2 = tf.get_variable("weight_2",[16,32])
result_2 = tf.nn.relu(tf.matmul(tf.reshape(support_2,[-1,16]),weight_2))

result_2 = tf.reshape(result_2,[-1,12,32])

flatten = tf.layers.Flatten()(supports)
dense_1 = tf.layers.Dense(units=256,activation='relu',use_bias=True)(flatten)
dense_2 = tf.layers.Dense(units=32,activation='relu',use_bias=True)(dense_1)
dense_3 = tf.layers.Dense(units=16,activation='relu',use_bias=True)(dense_2)
accuracy = tf.layers.Dense(units=1)(dense_3)


loss = tf.losses.mean_squared_error(labels=labels,predictions=accuracy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    support,feature,label = get_shuffled_data()
    losses = []
    val_losses = []

    epoch = 100
    batch_num = 50
    batch_size = 64
    for i in range(epoch):
        for j in range(batch_num):
            low = batch_size*j
            high = batch_size*(j+1)

            _,losss = sess.run((train,loss),
                    feed_dict={supports:support[low:high],features:feature[low:high],labels:label[low:high]})
            #val_loss = sess.run(loss,feed_dict={supports:support[400:],features:feature[400:],labels:label[400:]})
            losses.append(losss)
            #val_losses.append(val_loss)
            #print("mean_square_loss:",losss,"validation loss", val_loss)
            print("mean_square_loss:",losss)

    
        
    acc = sess.run(accuracy,feed_dict={supports:support[:300],features:feature[:300]})
    losses = np.asarray(losses)
    #val_losses = np.asarray(val_losses)
    #plt.plot(losses)
    plt.plot(label[:300])
    plt.plot(acc)
    #plt.plot(val_losses)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()