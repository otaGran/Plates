import numpy as np
import cv2
import glob as gb
import deskew
from PIL import Image
import matplotlib.pyplot as plt
import os
import string

showDebug = True
esp = 8

def parallel(cols,leftyA,leftyB,rightyA,rightyB):

    deltaX = abs(0 - (cols - 1))
    deltaY1 = (leftyB - 30) - (rightyB - 30)
    deltaY2 = (leftyA - 30) - (rightyA - 30)
    #print(deltaY1,deltaY2)

    M1 = deltaY1/deltaX
    M2 = deltaY2/deltaX
    #print("+++++++++++++++++++")

    #print(abs(deltaY2 - deltaY1))
    if abs(deltaY2 - deltaY1) > esp:
        return False
    else:
        return True

def getM(deltaX,deltaY):
    return deltaX / deltaY

def fitLine_ransac(pts, zero_add=0):
    if len(pts) >= 2:
        [vx, vy, x, y] = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01)
        lefty = int((-x * vy / vx) + y)
        righty = int(((136 - x) * vy / vx) + y)
        return lefty + 30 + zero_add, righty + 30 + zero_add
    return 0, 0


def findContoursAndDrawBoundingBox(image_rgb):
    image_rgb = cv2.resize(image_rgb, (140, 60))
    drawLine = image_rgb.copy()
    line_upper = [];
    line_lower = [];

    line_experiment = []
    grouped_rects = []
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

    # for k in np.linspace(-1.5, -0.2,10):
    for k in np.linspace(0,50, 16):
        drawRect = image_rgb.copy()
        drawRect = cv2.resize(drawRect,(0,0),fx=5,fy=5)
        # thresh_niblack = threshold_niblack(gray_image, window_size=21, k=k)
        # binary_niblack = gray_image > thresh_niblack
        # binary_niblack = binary_niblack.astype(np.uint8) * 255

        binary_niblack = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 17,
                                               k)

        tmp = binary_niblack.copy()
        tmp = cv2.resize(tmp, (0, 0), fx=5, fy=5)
        #cv2.imshow("image1",tmp)
        #cv2.waitKey(0)
        imagex, contours, hierarchy = cv2.findContours(binary_niblack.copy(), cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            bdbox = cv2.boundingRect(contour)
            #print((bdbox[3] / float(bdbox[2]), (bdbox[3] * bdbox[2])))

            if ((bdbox[3] / float(bdbox[2]) > 1.2 and bdbox[3] * bdbox[2] > 100 ) or (
                    bdbox[3] / float(bdbox[2]) > 3 and bdbox[3] * bdbox[2] < 150 and bdbox[3] * bdbox[2] > 50)):
                cv2.rectangle(drawRect, (bdbox[0]*5, bdbox[1]*5), (bdbox[0]*5 + bdbox[2]*5, bdbox[1]*5 + bdbox[3]*5), (255, 0, 0),
                              1)
                cv2.putText(drawRect, "*"+str(round(bdbox[3] / float(bdbox[2]),2)) +","+str(bdbox[3] * bdbox[2]),(bdbox[0]*5, bdbox[1]*5),cv2.FONT_HERSHEY_PLAIN,
  0.6, (0, 255, 255), 1, cv2.LINE_AA)

                line_upper.append([bdbox[0], bdbox[1]])
                line_lower.append([bdbox[0] + bdbox[2], bdbox[1] + bdbox[3]])
                # draw image
                drawLine = cv2.circle(drawLine, (bdbox[0], bdbox[1]), 1, (0, 255, 0), 1)
                drawLine = cv2.circle(drawLine, (bdbox[0] + bdbox[2], bdbox[1] + bdbox[3]), 1, (0, 0, 255), 1)

                line_experiment.append([bdbox[0], bdbox[1]])
                line_experiment.append([bdbox[0] + bdbox[2], bdbox[1] + bdbox[3]])
                # grouped_rects.append(bdbox)

    rgb = cv2.copyMakeBorder(image_rgb, 30, 30, 0, 0, cv2.BORDER_REPLICATE)
    leftyA, rightyA = fitLine_ransac(np.array(line_lower), 1)
    rows, cols = rgb.shape[:2]

    # rgb = cv2.line(rgb, (cols - 1, rightyA), (0, leftyA), (0, 0, 255), 1,cv2.LINE_AA)

    leftyB, rightyB = fitLine_ransac(np.array(line_upper), -1)

    rows, cols = rgb.shape[:2]

    # rgb = cv2.line(rgb, (cols - 1, rightyB), (0, leftyB), (0,255, 0), 1,cv2.LINE_AA)
    pts_map1 = np.float32([[cols - 1, rightyA], [0, leftyA], [cols - 1, rightyB], [0, leftyB]])
    pts_map2 = np.float32([[136, 36], [0, 36], [136, 0], [0, 0]])
    mat = cv2.getPerspectiveTransform(pts_map1, pts_map2)
    image = cv2.warpPerspective(rgb, mat, (136, 36), flags=cv2.INTER_CUBIC)
    #
    image, M = deskew.fastDeskew(image)

    try:
        cv2.line(drawLine, (cols - 1, rightyB - 30), (0, leftyB - 30), (255, 0, 0), 1, 16, 0)
        cv2.line(drawLine, (cols - 1, rightyA - 30), (0, leftyA - 30), (255, 0, 0), 1, 16, 0)
    except:
        return image, -1
    drawLine = cv2.resize(drawLine, (0, 0), fx=5, fy=5)
    if showDebug:
        cv2.imshow("drawLine", drawLine)
    if parallel(cols,leftyA,leftyB,rightyA,rightyB):
        return image, 0
    else:
        return image, -1


'''
# Returns a list of all folders with participant numbers
img_path = gb.glob("D:\\1070129\\*.jpg")
for path in img_path:
    img = cv2.imread(path)
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    print(path, "%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # Select ROI
    showCrosshair = False
    fromCenter = False
    r = cv2.selectROI("Image", img, fromCenter, showCrosshair)

    # Crop image
    imCrop = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
'''
"""
list_dirs = os.walk("/Users/beans/Desktop/AllPlatesVal")
for root, dirs, files in list_dirs:
    for f in files:
        print(os.path.join(root, f))
    #fn = '/Users/beans/Desktop/屏幕快照 2018-12-09 下午8.26.04.png'
        img = cv2.imread(os.path.join(root, f))
        img = findContoursAndDrawBoundingBox(img)
        cv2.waitKey(0)
"""