
import matplotlib.pyplot as plt
import numpy as np
import random

import string

characters = string.digits + string.ascii_uppercase + "-" + " "
rootdir = "6Plates"
previousModelPath = "/Users/beans/PycharmProjects/Plates/plates_model.h5"
# all digits and alph with dash and space
#print(characters)

batch_size, height, width, i, n_len, n_class = 10911 + 8473, 79, 180, 0, 6 + 1 + 1, len(characters)
# batch_size : 6Plates + 7 Plates
# n_len : 7 digits + 1(dash in the middle)

print(characters)



def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


from keras.models import *
from keras.layers import *

rnn_size = 128

input_tensor = Input((width, height, 3))
x = input_tensor
for i in range(3):
    x = Convolution2D(32, 3, 3, activation='relu')(x) #TODO padding == same
    x = Convolution2D(32, 3, 3, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

conv_shape = x.get_shape()
x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)

x = Dense(32, activation='relu')(x)

gru_1 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru1')(x)
gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru1_b')(x)
gru1_merged = merge([gru_1, gru_1b], mode='sum')

gru_2 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru2')(gru1_merged)
gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru2_b')(gru1_merged)
x = merge([gru_2, gru_2b], mode='concat')
x = Dropout(0.25)(x)
x = Dense(n_class, init='he_normal', activation='softmax')(x)
base_model = Model(input=input_tensor, output=x)

labels = Input(name='the_labels', shape=[n_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])

model = Model(input=[input_tensor, labels, input_length, label_length], output=[loss_out])
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')
# model.load_weights('model30.h5') //not working for custom loss fuctions


#model.load_model(previousModelPath, custom_objects={'<lambda>': lambda y_ture, y_pred: y_pred})

# [X_test, y_test, _, _], _ = next(gen(1))
# print(y_test.shape)
# print(y_test)
# print(X_test.shape)
# print(''.join([characters[x] for x in y_test[0]]))
# plt.imshow(X_test[0].transpose(1, 0, 2))
# plt.title(''.join([characters[x] for x in y_test[0]]))

import string

import os.path
import matplotlib.image as mpimg

# this folder is custom

#characters = string.digits + string.ascii_uppercase + "-" + " "

X = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
y = np.zeros((batch_size, n_len), dtype=np.uint8)
for parent, dirnames, filenames in os.walk(rootdir):
    # case 2
    for filename in filenames:
        if filename.endswith(".jpg"):
            X[i] = mpimg.imread(os.path.join(parent, filename)).transpose(1, 0, 2)
            random_str = filename[filename.find('_') + 1:].replace('.jpg', '') + " "

            # print(random_str)
            # print([characters.find(x) for x in random_str])
            y[i] = [characters.find(x) for x in random_str]
            # print(y[i])
            i += 1
            # print("full path" + os.path.join(parent,filename))

for parent, dirnames, filenames in os.walk("7Plates"):
    # case 2
    for filename in filenames:
        if filename.endswith(".jpg"):
            X[i] = mpimg.imread(os.path.join(parent, filename)).transpose(1, 0, 2)
            random_str = filename[filename.find('_') + 1:].replace('.jpg', '')
            if len(random_str) != 8:
                continue
            # print(random_str)
            # print([characters.find(x) for x in random_str])
            y[i] = [characters.find(x) for x in random_str]
            # print(y[i])
            i += 1
            # print("full path" + os.path.join(parent,filename))
print(i)
characters2 = characters + ' '
valdir = "6PlatesVal"
filenames = os.listdir(valdir)
X_test = np.zeros((197, width, height, 3), dtype=np.uint8)
y_test = np.zeros((197, n_len), dtype=np.uint8)
i = 0
for parent, dirnames, filenames in os.walk(valdir):
    # case 2
    for filename in filenames:
        if filename.endswith(".jpg"):
            X_test[i] = mpimg.imread(os.path.join(parent, filename)).transpose(1, 0, 2)
            random_str = filename[filename.find('_') + 1:].replace('.jpg', '')
            if len(random_str) != 8:
                continue
            print(random_str)
            # print([characters.find(x) for x in random_str])
            y_test[i] = [characters.find(x) for x in random_str]
            # print(y[i])
            i += 1
            # print("full path" + os.path.join(parent,filename))

"""
def gen(batch_size=1):
    X = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
    y = np.zeros((batch_size, n_len), dtype=np.uint8)
    i = 0
    while True:
        filename = filenames[i]
        # iprint(filename)
        # random_str = ''.join([random.choice(characters) for j in range(6)])
        random_str = filename[filename.find('_') + 1:].replace('.jpg', '') + " "
        X[0] = mpimg.imread(os.path.join(valdir, filename)).transpose(1, 0, 2)
        y[0] = [characters.find(x) for x in random_str]
        i += 1
        yield [X, y, np.ones(batch_size) * int(conv_shape[1] - 2), np.ones(batch_size) * n_len], np.ones(batch_size)

"""
def evaluate(model, test_num):
    batch_acc = 0
    #generator = gen()
    for i in range(test_num):
        # [X_test, y_test, _, _], _  = next(generator)
        y_pred = base_model.predict(X_test[i][np.newaxis, :])
        shape = y_pred[:, 2:, :].shape
        out = K.get_value(K.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0])[:, :8]
        if out.shape[1] == 8:
            batch_acc += ((y_test[i] == out).sum(axis=1) == 8).mean()
            argmax = np.argmax(y_pred, axis=2)[0]

    return batch_acc / test_num


from keras.callbacks import *


class Evaluate(Callback):
    def __init__(self):
        self.accs = []

    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate(base_model) * 100
        print()

        # todo single print not working!
        print(acc)
        print(acc)
        self.accs.append(acc)
        print(acc)
        print("sss")



evaluator = Evaluate()
evaluator.on_epoch_end(1)
# TODO change API from keras 1 to keras2.x
# TODO Keras shuffle after split!!!!
# TODO Auto save stage model with the name of acc
model.fit([X, y, np.ones(batch_size) * int(conv_shape[1] - 2), np.ones(batch_size) * n_len], np.ones(batch_size),
          batch_size=16, nb_epoch=200, callbacks=[EarlyStopping(patience=10), evaluator], validation_split=0.1,
          shuffle=True)

model.save('modelcallback.h5')
