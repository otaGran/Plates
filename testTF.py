import tensorflow as tf
import os
import numpy as np
import random
import datetime
import string
import time
import matplotlib.image as mpimg
from keras import backend as K

characters = string.digits + string.ascii_uppercase + "-" + " "
val_len = 200
batch_size, height, width, i, n_len, n_class = 19652, 80, 170, 0, 6 + 1 +1 , len(characters)
valdir = "AllPlatesVal17080"
filenames = os.listdir(valdir)
X_test = np.zeros((val_len, width, height, 3), dtype=np.uint8)
y_test = np.zeros((val_len, n_len), dtype=np.uint8)
i = 0
for parent, dirnames, filenames in os.walk(valdir):
    # case 2
    for filename in filenames:
        if filename.endswith(".jpg"):
            X_test[i] = mpimg.imread(os.path.join(parent, filename)).transpose(1, 0, 2)
            random_str = filename[filename.find('_') + 1:].replace('.jpg', '')
            if len(random_str) == 7:
                random_str += " "
            print(random_str+"-")
            #print([characters.find(x) for x in random_str])
            y_test[i] = [characters.find(x) for x in random_str]
            #print(y[i])
            i += 1
            # print("full path" + os.path.join(parent,filename))

def evaluate(test_model):
    batch_num=200
    batch_acc = 0
    true_acc = 0
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    print(st)
    for i in range(batch_num):
        #[X_test, y_test, _, _], _  = next(generator)
        y_pred = test_model.predict(X_test[i][np.newaxis,:])
        shape = y_pred[:,2:,:].shape
        out = K.get_value(K.ctc_decode(y_pred[:,2:,:], input_length=np.ones(shape[0])*shape[1])[0][0])[:, :8]
        #if out.shape[1] == 8:
            #batch_acc += ((y_test[i] == out).sum(axis=1) == 8).mean()
            #argmax = np.argmax(y_pred, axis=2)[0]
        out = ''.join([characters[x] for x in out[0]]).replace(' ','')
        y_true = ''.join([characters[x] for x in y_test[i]]).replace(' ','')
        if out == y_true:
            true_acc += 1
        """
        else:
            print(out)
            print(y_true)
            print("-----------")
        """
    #print(true_acc / batch_num*100)
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    print(st)
    return true_acc / batch_num*100



def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

import argparse
import tensorflow as tf

if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="modelcallback.pb", type=str, help="Frozen model file to import")
    args = parser.parse_args()

    # We use our "load_graph" function
    graph = load_graph(args.frozen_model_filename)

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)
        for j in op.outputs:
            print(j.get_shape())
        print()
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions

    # We access the input and output nodes v
    x = graph.get_tensor_by_name('prefix/input_1:0')
    y = graph.get_tensor_by_name('prefix/Exp:0')
    z = graph.get_tensor_by_name('prefix/the_labels:0')
    label_length = graph.get_tensor_by_name('prefix/label_length:0')
    input_length = graph.get_tensor_by_name('prefix/input_length:0')

    # We launch a Session
    with tf.Session(graph=graph) as sess:
        # Note: we don't nee to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants
        #W y_out =
        # I taught a neural net to recognise when a sum of numbers is bigger than 45
        # it should return False in this case
        #print(y_out.shape)  # [[ False ]] Yay, it works!
        #shape = y_out[:, 2:, :].shape
        #out = K.get_value(K.ctc_decode(y_out[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0])[:, :8]
        # if out.shape[1] == 8:
        # batch_acc += ((y_test[i] == out).sum(axis=1) == 8).mean()
        # argmax = np.argmax(y_pred, axis=2)[0]
        #out = ''.join([characters[x] for x in out[0]]).replace(' ', '')
        #print(out)

        batch_num = 1#264
        batch_acc = 0
        true_acc = 0
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        print(st)
        print(datetime.datetime.now())
        for i in range(batch_num):
            # [X_test, y_test, _, _], _  = next(generator)
            print(X_test[i])
            y_pred = sess.run(y, feed_dict={
            x:X_test[i][np.newaxis, :]


        })
            shape = y_pred[:, 2:, :].shape
            out = K.get_value(K.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0])[:, :8]
            # if out.shape[1] == 8:
            # batch_acc += ((y_test[i] == out).sum(axis=1) == 8).mean()
            # argmax = np.argmax(y_pred, axis=2)[0]
            out = ''.join([characters[x] for x in out[0]]).replace(' ', '')
            y_true = ''.join([characters[x] for x in y_test[i]]).replace(' ', '')
            if out == y_true:
                true_acc += 1

            """
            else:
                print(out)
                print(y_true)
                print("-----------")
            """
        # print(true_acc / batch_num*100)
        print(datetime.datetime.now())
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        print(st)
        print(true_acc / batch_num * 100)
