import tensorflow as tf
import numpy as np
import json
import os
import matplotlib.pyplot as plt

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

tf.set_random_seed(777)  # reproducibility

json_data = open("input.txt").read()
json_data2 = json.loads(json_data)
data = []

for index, _data in enumerate(json_data2):
    data.append([_data['high'], _data['low'], _data['open'], _data['close'], _data['volume'], _data['weightedAverage'], 0])

for i in range(1, len(data)-1-12):
    for j in range(1, 13):
        data[i][6] += data[i+j][5]
    data[i][6] /= 12
    #data[i][6] /= data[i][3]

    #if(data[i][6]>=1):
    #    data[i][6]=1
    #else:
    #    data[i][6]=0

#data = np.copy(data)

'''for i in range(len(data)-1, 0, -1):
    for j in range(len(data[i])):
        if(data[i][j]!=0 and data[i-1][j]!=0):
            data[i][j]/=data[i-1][j]
        else:
            data[i][j]=1'''


data.pop(0)
for i in range(0, 13):
    data.pop(len(data)-1)

#print(data[0], data[1], data[2], data[3], data[len(data)-2], data[len(data)-1])

seq_length = 6 # 몇개볼건지
data_dim = 6 # 인풋데이터종류수
hidden_dim = 12
output_dim = 1
learning_rate = 0.001
iterations = 10

data = MinMaxScaler(data)

data = np.copy(data)

#print(np.array(data).shape)
x = data[:, :-1]
y = data[:, [-1]]

#print(y)

dataX = []
dataY = []

for i in range(0, len(y) - seq_length):
    _x = x[i:i+seq_length]
    _y = y[i+seq_length]
    #print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

train_size = int(len(dataY)*0.85)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])

X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])
lr = tf.placeholder(tf.float32)

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
#cell = tf.contrib.rnn.BasicLSTMCell(num_units=1, state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output



# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        '''if i < 100:
            forlr = 0.01
        elif i < 200:
            forlr = 0.003
        elif i < 300:
            forlr = 0.001
        else:
            forlr = 0.0001'''
        _, step_loss = sess.run([train, loss], feed_dict={
                                X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(i, step_loss))

    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    print(np.shape(test_predict))
    print(np.shape(testX))
    print(np.shape(testY))
    result_list = []
    result2_list = []
    for i in range(len(testX)):
        result = test_predict[i] / testX[i][5][5]
        #print(result)
        result2 = testY[i] / testX[i][5][5]
        #print(result2)
        if result >= 1.05:
            result_list.append(1)
        elif result <= 0.95:
            result_list.append(0)
        else:
            result_list.append(0.5)

        if result2 >= 1.05:
            result2_list.append(1)
        elif result2 <= 0.95:
            result2_list.append(0)
        else:
            result2_list.append(0.5)

    a,b,c,d = 0,0,0,0
    for i in range(len(testY)):
        if result_list[i] == result2_list[i]:
            a+=1
            if result_list[i] == 1:
                c+=1
            elif result_list[i] == 0:
                d+=1
        else:
            b+=1



    rmse_val = sess.run(rmse, feed_dict={
                    targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))

    print("a : ", a, "\tb : ", b)
    print("떡상 : ", c, "\t떡락 : " , d)


    # Plot predictions
    plt.plot(testY)
    plt.plot(test_predict)
    plt.plot(result_list)
    plt.plot(result2_list)
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.show()
