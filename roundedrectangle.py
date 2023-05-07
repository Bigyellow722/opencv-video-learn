import os
import cv2
import numpy as np
import math
from enum import Enum
import matplotlib.pyplot as plt

class Degree(Enum):
    DEGREE_25 = 0
    DEGREE_40 = 1
    DEGREE_50 = 2
    DEGREE_60 = 3


rgb_counter = 0;
    
def showImgInNewWindow(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name, img)

def showContours(img, contours):
    img_contours = np.copy(img)
    cv2.drawContours(img_contours, contours, -1, (0, 0, 0), 2)
    showImgInNewWindow("Contours", img_contours)

def denoise(gray, method):
    """灰度图的去噪。

    Args:
        gray (np.ndarray): 待去噪的灰度图。
        method (str): 去噪所使用的方法名。

    Returns:
        blurred: 去噪后得到的图像。
    """
    if method == "MedianBlur":
        blurred = cv2.medianBlur(gray, 5)
        showImgInNewWindow("MedianBlur", blurred)
        return blurred
    elif method == "GuassBlur":
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        showImgInNewWindow("GuassBlur", blurred)
        return blurred
    else:
        print("No such denoise method!")
        return np.copy(gray)


def binarize(gray, method):
    """灰度图的二值化。

    Args:
        gray (np.ndarray): 待二值化的灰度图。
        method (str): 二值化所使用的方法名。

    Returns:
        thresh: 二值化后得到的图像。
    """
    if method == "AdaptiveThreshold":
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 1)
        showImgInNewWindow("AdaptiveThreshold", thresh)
        return thresh
    elif method == "Canny":
        thresh = cv2.Canny(gray, 50, 100)
        showImgInNewWindow("Canny", thresh)
        return thresh
    else:
        print("No such binarize method!")
        return np.copy(gray)


def filterRepeatedContours(contours, centroids):
    """去除contours与cnetroids中重复的轮廓与相应的中心。

    Args:
        contours (list): 待去重的轮廓的列表。
        centroids (list): 待去重的轮廓中心的列表。

    Returns:
        contours: 去除重复轮廓后的轮廓列表。
        centroids: 去除重复轮廓对应的中心后的轮廓中心列表。
    """
    # 调试：输出所有轮廓的相关数据
    # for cid in range(len(contours)):
    #     print("Contour: %d" % cid)
    #     print("Vertex Count: {0}".format(contours[cid].shape[0]))
    #     print("Centroid: {0}".format(centroids[cid]))
    #     print("Area: {0}".format(cv2.contourArea(contours[cid])))
    #     print()

    # 使用 [1.中点距离, 2.形状相似度, 3. 面积差值] 三个条件来判断轮廓是否重复
    is_valid = np.ones(len(contours), dtype=bool)
    area = [cv2.contourArea(c) for c in contours]
    for cid0 in range(len(contours)):
        if is_valid[cid0]:
            area0 = area[cid0]
            for cid1 in range(cid0+1, len(contours)):
                vec = centroids[cid0] - centroids[cid1]
                distance = np.linalg.norm(vec)
                area1 = area[cid1]
                area_diff = math.fabs(area0-area1)
                match = cv2.matchShapes(contours[cid0], contours[cid1], 1, 0.0)
                if distance < 30 and area_diff < 1000 and match < 0.03:
                    is_valid[cid1] = False
    contours = [contours[cid] for cid in range(len(contours)) if is_valid[cid]]
    centroids = [centroids[cid] for cid in range(len(centroids)) if is_valid[cid]]

    # 返回筛选过后的轮廓及其中心
    return contours, centroids

def getPosHSV(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(param[y, x])

def maskImage(hsv_image, degree):
    if degree == Degree.DEGREE_25:
        #25-degree:
        hsv_low = np.array([80, 30, 55])
        hsv_up = np.array([175, 220, 200])
    elif degree == Degree.DEGREE_40:
        #40-degree:
        hsv_low = np.array([70, 30, 55])
        hsv_up = np.array([175, 260, 210])
    elif degree == Degree.DEGREE_50:
        #50-degree:
        hsv_low = np.array([40, 40, 55])
        hsv_up = np.array([175, 230, 160])
    elif degree == Degree.DEGREE_60:
        #60-degree:
        hsv_low = np.array([15, 60, 55])
        hsv_up = np.array([150, 260, 150])
    else:
        hsv_low = np.array([40, 40, 55])
        hsv_up = np.array([175, 230, 160])

    mask = cv2.inRange(hsv_image, hsv_low, hsv_up);
    output = cv2.bitwise_and(hsv_image, hsv_image, mask = mask)
    bgr_output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
    showImgInNewWindow("mask", bgr_output)
    return bgr_output


b_list = []
g_list = []
r_list = []


def drawBgrPlt(img):
    global rgb_counter
    rgb_counter += 1
    img_b = img[:,:,0]
    img_g = img[:,:,1]
    img_r = img[:,:,2]
    img_b_np = np.array(img_b)
    img_g_np = np.array(img_g)
    img_r_np = np.array(img_r)
    b_mean = np.mean(img_b_np)
    g_mean = np.mean(img_g_np)
    r_mean = np.mean(img_r_np)
    print("test", b_mean)
    b_list.append(b_mean)
    g_list.append(g_mean)
    r_list.append(r_mean)



def shapeDetect(img_path, degree):
    img = cv2.imread(img_path)

    print(img.shape)
    showImgInNewWindow("Original", img)

    '''
    25-degree: x:500-880, y:200-440
    50-degree: x:500-980, y:200-440
    '''
    if degree == Degree.DEGREE_25:
        cropped_img = img[200:440, 500:880]
    elif degree == Degree.DEGREE_40:
        cropped_img = img[200:440, 500:980]
    elif degree == Degree.DEGREE_50:
        cropped_img = img[200:440, 500:980]
    elif degree == Degree.DEGREE_60:
        cropped_img = img[200:500, 500:980]
    else:
        cropped_img = img

    showImgInNewWindow("CROPPED", cropped_img)


    blurred = cv2.GaussianBlur(cropped_img, (5, 5), 0)

    showImgInNewWindow("BLURRED", blurred)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    showImgInNewWindow("HSV", hsv)
    cv2.setMouseCallback("HSV", getPosHSV, hsv)

    mask_img = maskImage(hsv, degree)

    gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

    # 转换为灰度图
    showImgInNewWindow("Gray", gray)

    # 去噪
    denoised = denoise(gray, "MedianBlur")
    thresh = binarize(denoised, "Canny")

    #binary = cv2.bitwise_not(gray)

    #contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 计算轮廓的质心
    centroids = []
    for c in contours:
        mu = cv2.moments(c, False)
        if np.isclose(mu['m00'], 0):
            mc = contours[0][0]
        else:
            mc = [mu['m10'] / mu['m00'], mu['m01'] / mu['m00']]
        mc = np.intp(mc)
        centroids.append(mc)

    # 去除重复的轮廓
    contours, centroids = filterRepeatedContours(contours, centroids)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        area = w * h

        if degree == Degree.DEGREE_25:
            area_list = [32000, 50000]
        elif degree == Degree.DEGREE_40:
            area_list = [32000, 50000]
        elif degree == Degree.DEGREE_50:
            area_list = [32000, 50000]
        elif degree == Degree.DEGREE_60:
            area_list = [32000, 50000]
        else:
            area_list = [32000, 50000]

        #global rgb_counter
        #print("shapeDetect", rgb_counter);
        if (area > area_list[0] and area < area_list[1]):
            #print("x:", x, "y:", y, "w:", w, "h:", h)
            rect_img = cv2.rectangle(cropped_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #cv2.imshow("rect", rect_img)
            rect_out = cropped_img[y:y+h, x:x+w]
            #cv2.imshow("rect_out", rect_out)
            drawBgrPlt(rect_out)


def sortPng(filename):
    filenum = filename[-8:-4]
    #print(filenum)
    return int(filenum)


def findAllPng(directory):
    #for path in path_list:
    #    print(path)
    path_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.png'):
                path_list.append(os.path.join(root, file))
                #print(os.path.join(root, file))
    path_list.sort(key=sortPng)
    #print(path_list)
    return path_list

def drawPlot():
    b_x = range(len(b_list))
    g_x = range(len(g_list))
    r_x = range(len(r_list))
    plt.plot(b_x, b_list, '-',color='blue')
    plt.plot(g_x, g_list, '-',color='green')
    plt.plot(r_x, r_list, '-',color='red')
    #give the name of the x and y  axis
    plt.xlabel('x label')
    plt.ylabel('y label')
    #also give the title of the plot
    plt.title("Title")
    plt.show()


def main():

    '''
    degree_25 = '/home/wqy/rasp_space/videos/25-degree'
    file_list = findAllPng(degree_25)
    for img_path in file_list:
        print(img_path)
        shapeDetect(img_path, Degree.DEGREE_25)
    print(b_list)
    print(g_list)
    print(r_list)
    print('\n')
    drawPlot()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    img_path = '/home/wqy/rasp_space/videos/40-degree/0001.png'

    global rgb_counter
    print("before shapeDetect", rgb_counter);
    shapeDetect(img_path, Degree.DEGREE_40)
    print("after shapeDetect", rgb_counter);
    print(b_list)
    print(g_list)
    print(r_list)
    print('\n')

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
