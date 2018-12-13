import tensorflow as tf
import os
import numpy as np

import datetime
import string
import time
from keras import backend as K
import fineMapping_Old



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

    emptyImage3 = np.zeros((80, 170), np.uint8)

    area = 0
    #print(dst.shape)

    height, width = dst.shape
    cutFlag = True
    cutPos_left = 0
    cutPos_right = 170
    offset_left = 80 - 50
    offset_right = 80 - 50
    for j in range(width):
        area = 0
        for i in range(height):
            if dst[i, j] == 0:
                area += 1
        if (area < offset_left) and cutFlag:
            #print(area)
            cutPos_left = j
            cutFlag = False
        if area < offset_right:
            cutPos_right = j
        cv.line(emptyImage3, (j, 80), (j, area), (255, 255, 255))
    #cv.imshow("sobel_hist", emptyImage3)
    #cv.imshow("sobel_threshold", dst)
    #print(cutPos_left)
    #print(cutPos_right)
    return_value = 0
    if cutPos_right - cutPos_left < 50:
        return dst,-1
    else:
        dst = image[0:80, cutPos_left:cutPos_right]
        #cv.imshow("fineMapping_hor", dst)
        return dst,return_value


characters = string.digits + string.ascii_uppercase + "-" + " "
val_len = 200
batch_size, height, width, i, n_len, n_class = 19652, 80, 170, 0, 6 + 1 + 1, len(characters)
X_test = np.zeros((val_len, width, height, 3), dtype=np.uint8)
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






import numpy as np
import cv2 as cv
import math

face_cascade = cv.CascadeClassifier('/Users/beans/Downloads/cascade15*best.xml')
parser = argparse.ArgumentParser()
parser.add_argument("--frozen_model_filename", default="modelcallback.pb", type=str,
                    help="Frozen model file to import")
args = parser.parse_args()

# We use our "load_graph" function
graph = load_graph(args.frozen_model_filename)

x = graph.get_tensor_by_name('prefix/input_1:0')
y = graph.get_tensor_by_name('prefix/Exp:0')
with tf.Session(graph=graph) as sess:
    list_dirs = os.walk("/Users/beans/Desktop/final/tes/")
    for root, dirs, files in list_dirs:
        for f in files:
            if f.endswith('jpg'):
                frame = cv.imread(os.path.join(root, f))
                ori_height, ori_width,channel =frame.shape
                #print(ori_height)
                #print(ori_width)
                ori_offset = float(ori_height * ori_width) / float(800 * 800)
                #print(ori_offset)
                #print(math.sqrt(ori_offset))
                #print((1.0/math.sqrt(ori_offset)))
                frame = cv.resize(frame,(0,0),fx=1.0/(math.sqrt(ori_offset)),fy=1.0/(math.sqrt(ori_offset)))
                #print(frame.shape)
            else:
                continue
            #print(os.path.join(root, f))
            drawLRect = frame.copy()
            faces = face_cascade.detectMultiScale(frame, 1.2, 4)
            for (A, B, w, h) in faces:
                validPlate = calSobel(frame[B:B + h, A:A + w])
                if (validPlate):
                    expand = 4
                    if B - int((h / expand)) < 0:
                        true_B = B
                    else:
                        true_B = B - int((h / expand))
                    if A - int((w / expand)) < 0:
                        true_A = A
                    else:
                        true_A = A - int((w / expand))
                    roi_color = frame[true_B:B + int(h + (h / expand)), true_A:A + int(w + (w / expand))]
                    roi_color, code = fineMapping_Old.findContoursAndDrawBoundingBox(roi_color)
                    roi_color = cv.resize(roi_color, (170, 80))
                    #cv.imshow("ori",roi_color)
                    if code == -1:
                        continue
                    roi_color, code  = drawSobelVertical(roi_color)
                    if code == -1:
                        continue
                    roi_color = cv.resize(roi_color, (170, 80))
                    tests = roi_color.copy()
                    X_test[0] = roi_color.transpose(1, 0, 2)
                    y_pred = sess.run(y, feed_dict={
                        x: X_test[0][np.newaxis, :]
                    })
                    shape = y_pred[:, 2:, :].shape
                    out = K.get_value(
                        K.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0])[
                          :, :8]
                    out = ''.join([characters[x] for x in out[0]]).replace(' ', '')
                    print(out)
                    # TODO return multi plates in the same time
                    a = cv.copyMakeBorder(tests, 0, 80, 0, 0, cv.BORDER_CONSTANT, value=[237, 237, 237])
                    cv.putText(a, out, (0, 140), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1, cv.LINE_AA)
                    #cv.imshow("tests", a)



                    """
                    roi_color = drawSobelVertical(roi_color)

                    roi_color = cv.resize(roi_color, (170, 80))
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
                    """

                    cv.rectangle(drawLRect, (A, B), (A + w, B + h), (255, 0, 0), 2)
                else:
                    cv.rectangle(drawLRect, (A, B), (A + w, B + h), (0, 0, 255), 2)
            #cv.imshow('frame', drawLRect)
            cv.waitKey(0)
cv.destroyAllWindows()
