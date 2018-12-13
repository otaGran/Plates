import tensorflow as tf
import os
import numpy as np
import random
import datetime
import string
import time
import matplotlib.image as mpimg
from keras import backend as K
import fineMapping_Old
from matplotlib import pyplot as plt

def calSobel(image):
    # Mat

    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (170, 80))
    img = cv.GaussianBlur(img, (5, 5), 0)
    x = cv.Sobel(img, cv.CV_16S, 1, 0)
    y = cv.Sobel(img, cv.CV_16S, 0, 1)
    absX = cv.convertScaleAbs(x)  # 转回uint8
    absY = cv.convertScaleAbs(y)

    dst = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
    rret, dst = cv.threshold(dst, 80, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    area = 0
    height, width = dst.shape
    for i in range(height):
        for j in range(width):
            if dst[i, j] == 255:
                area += 1
    # print(area)

    # cv.imshow("ori",img)
    # cv.imshow("thre",dst)
    # cv.waitKey(0)
    if area > 4000:
        return True
    else:
        return False

def drawSobelVertical(image):
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (170, 80))
    img = cv.GaussianBlur(img, (5, 5), 0)
    x = cv.Sobel(img, cv.CV_16S, 1, 0)
    y = cv.Sobel(img, cv.CV_16S, 0, 1)
    absX = cv.convertScaleAbs(x)  # 转回uint8
    absY = cv.convertScaleAbs(y)

    dst = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
    rret, dst = cv.threshold(dst, 80, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    area = 0
    height, width = dst.shape

    emptyImage3 = np.zeros((80,170), np.uint8)

    area = 0
    print(dst.shape)

    height, width = dst.shape
    cutFlag = True
    cutPos = 0
    cutPos2 = 0
    for j in range(width):
        area = 0
        for i in range(height):
            if dst[i, j] == 0:
                area += 1
        if (area < 80 - 60) and cutFlag:
            print(area)
            cutPos = j
            cutFlag = False
        if area < 80 - 50:
            cutPos2 = j
        cv.line(emptyImage3,(j,80),(j,area),(255,255,255))
    cv.imshow("empty", emptyImage3)
    cv.imshow("thre",dst)
    dst = image[0:80, cutPos:cutPos2]
    cv.imshow("cut", dst)

    return dst




characters = string.digits + string.ascii_uppercase + "-" + " "
val_len = 200
batch_size, height, width, i, n_len, n_class = 19652, 80, 170, 0, 6 + 1 + 1, len(characters)
valdir = "AllPlatesVal17080"
filenames = os.listdir(valdir)
X_test = np.zeros((val_len, width, height, 3), dtype=np.uint8)
y_test = np.zeros((val_len, n_len), dtype=np.uint8)
i = 0


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
    parser.add_argument("--frozen_model_filename", default="modelcallback.pb", type=str,
                        help="Frozen model file to import")
    args = parser.parse_args()

    # We use our "load_graph" function
    graph = load_graph(args.frozen_model_filename)

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        #print(op.name)
        for j in op.outputs:
            print(j.get_shape())
        #print()
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

import numpy as np
import cv2 as cv

face_cascade = cv.CascadeClassifier('/Users/beans/Downloads/cascade15*best.xml')
# eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
cap = cv.VideoCapture('480p.mov')

# We launch a Session
cnt = 0
with tf.Session(graph=graph) as sess:
    while (cap.isOpened()):
        cnt += 1
        ret, frame = cap.read()


        drawLRect = frame.copy()

        #cv.imshow("fuck", frame)
        # frame = cv.imread("/Users/beans/Desktop/final/tes/13771544415122_.pic.jpg")

        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # img = cv.imread('test.jpg')
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if cnt % 5 == 0:
            faces = face_cascade.detectMultiScale(frame, 1.1, 5)

            for (A, B, w, h) in faces:
                validPlate = calSobel(frame[B:B + h, A:A + w])
                if (validPlate):
                    if cnt % 25 == 0:
                        if B - int((h / 4)) < 0:
                            true_B = 0
                        else:
                            true_B = B - int((h / 4))

                        if A - int((w / 4)) < 0:
                            true_A = 0
                        else:
                            true_A = A - int((w / 4))

                        roi_color = frame[true_B:B + int(h + (h / 4)), true_A:A + int(w + (w / 4))]
                        if A < 10:
                            true_A = 0
                        else:
                            true_A = A-10
                        #roi_color = frame[B:B+h, true_A:A +w]
                        roi_color, code = fineMapping_Old.findContoursAndDrawBoundingBox(roi_color)
                        #print(roi_color.shape)

                        if code == -1:
                            continue
                        roi_color = cv.resize(roi_color, (170, 80))
                        tests = roi_color.copy()


                        X_test[0] = roi_color.transpose(1, 0, 2)
                        # print(datetime.datetime.now())
                        #
                        y_pred = sess.run(y, feed_dict={
                            x: X_test[0][np.newaxis, :]

                        })
                        shape = y_pred[:, 2:, :].shape
                        out = K.get_value(
                            K.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0])[
                              :, :8]
                        # if out.shape[1] == 8:
                        # batch_acc += ((y_test[i] == out).sum(axis=1) == 8).mean()
                        # argmax = np.argmax(y_pred, axis=2)[0]
                        out = ''.join([characters[x] for x in out[0]]).replace(' ', '')
                        print(out)

                        a = cv.copyMakeBorder(tests, 0, 80, 0, 0, cv.BORDER_CONSTANT, value=[237, 237, 237])
                        cv.putText(a, out, (0,140), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1, cv.LINE_AA)

                        cv.imshow("tests", a)

                        ###########
                        roi_color = drawSobelVertical(roi_color)
                        roi_color = cv.resize(roi_color,(170,80))
                        X_test[0] = roi_color.transpose(1, 0, 2)
                        # print(datetime.datetime.now())
                        #
                        y_pred = sess.run(y, feed_dict={
                            x: X_test[0][np.newaxis, :]

                        })
                        shape = y_pred[:, 2:, :].shape
                        out = K.get_value(
                            K.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0])[
                              :, :8]
                        # if out.shape[1] == 8:
                        # batch_acc += ((y_test[i] == out).sum(axis=1) == 8).mean()
                        # argmax = np.argmax(y_pred, axis=2)[0]
                        out = ''.join([characters[x] for x in out[0]]).replace(' ', '')
                        print(out)

                        a = cv.copyMakeBorder(roi_color, 0, 80, 0, 0, cv.BORDER_CONSTANT, value=[237, 237, 237])
                        cv.putText(a, out, (0, 140), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1, cv.LINE_AA)

                        cv.imshow("testCut", a)
                        cv.waitKey(1)
                    cv.rectangle(drawLRect, (A, B), (A + w, B + h), (255, 0, 0), 2)
                else:
                    cv.rectangle(drawLRect, (A, B), (A + w, B + h), (0, 0, 255), 2)
        cv.imshow('frame', drawLRect)
        cv.waitKey(1)

cap.release()
cv.destroyAllWindows()

# eyes = eye_cascade.detectMultiScale(roi_gray)
# for (ex,ey,ew,eh) in eyes:
#    cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
# cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()
