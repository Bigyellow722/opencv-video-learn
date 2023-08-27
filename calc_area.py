import os
import cv2
import numpy as np
import math
from enum import Enum
import matplotlib.pyplot as plt

def getPosHSV(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(param[y, x])


def showImgInNewWindow(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name, img)
    cv2.imwrite(name + "_85.png", img)


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

def maskImage(hsv_image):
    hsv_low = np.array([0, 0, 182])
    hsv_up = np.array([255, 255, 255])
    mask = cv2.inRange(hsv_image, hsv_low, hsv_up)
    output = cv2.bitwise_and(hsv_image, hsv_image, mask = mask)
    bgr_output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)

    return bgr_output

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

    
def calc_area_proportion(img_path):
    img = cv2.imread(img_path)
    showImgInNewWindow("Original", img)

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    showImgInNewWindow("BLURRED", blurred)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    showImgInNewWindow("HSV", hsv)
    cv2.setMouseCallback("HSV", getPosHSV, hsv)

    mask_img = maskImage(hsv)

    gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

    # 转换为灰度图
    showImgInNewWindow("Gray", gray)

    # 去噪
    denoised = denoise(gray, "MedianBlur")
    thresh = binarize(denoised, "Canny")

    binary = cv2.bitwise_not(thresh)
    showImgInNewWindow("Binary", binary)

    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    '''
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
    '''
    for contour in contours:
        area = cv2.contourArea(contour)
        print("contour area:", area)
        #if area > 100:
        cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

    showImgInNewWindow("Output", img)
    cv2.imwrite("output_85.png", img);


def main():
    img_name = '/home/wqy/rasp_space/videos/area/0085.png'
    calc_area_proportion(img_name)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
