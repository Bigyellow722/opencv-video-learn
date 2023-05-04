import cv2
import numpy as np
import math


class PolygonContainer:
    """存储检测到的特定类型的多边形的容器。

    Attributes:
        name (str): 该容器存储的多边形的类型名称。
        vertex_count (int): 该多边形类型的顶点数需满足的条件，若为0则代表任意值。
        check (func(contour, approx) -> bool): 该多边形类型的特判函数。
        contours (list): 检测到的该类型的多边形的轮廓的列表。
        centroids (list): 检测到的该类型的多边形的轮廓中心的列表。
    """

    def __init__(self, name, vertex_count, check):
        """PolygonContainer类的初始化函数，各参数含义与类注释中一致。
        """
        self.name = name
        self.vertex_count = vertex_count
        self.check = check
        self.contours = []
        self.centroids = []


def showPolygonContours(title, img, contours, centroids):
    """在img上绘制多边形并显示。

    Args:
        title (str): 显示窗口的标题。
        img (np.ndarray): 待绘制多边形的图像。
        contours (list): 多边形的轮廓列表。
        centroids (list): 多边形的轮廓中心列表。
    """
    # 在img上描绘出contours和centroids所指定的多边形，并进行显示
    img_temp = np.copy(img)
    cv2.drawContours(img_temp, contours, -1, (0, 0, 0), 3)
    for mc in centroids:
        cv2.circle(img_temp, tuple(mc), 2, (0, 0, 0), 5)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow(title, img_temp)


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
        cv2.namedWindow("MedianBlur", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow("MedianBlur", blurred)
        return blurred
    elif method == "GuassBlur":
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        cv2.namedWindow("GuassBlur", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow("GuassBlur", blurred)
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
        cv2.namedWindow("AdaptiveThreshold", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow("AdaptiveThreshold", thresh)
        return thresh
    elif method == "Canny":
        thresh = cv2.Canny(gray, 50, 100)
        cv2.namedWindow("Canny", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Canny", thresh)
        return thresh
    else:
        print("No such binarize method!")
        return np.copy(gray)


def rectangleCheck(contour, approx):
    """矩形特判函数。

    Args:
        contour (np.ndarray): 待判定的轮廓。
        approx (np.ndarray): 待判定的轮廓的多边形逼近。

    Returns:
        bool: 表示contour与approx是否表示一个矩形。
    """
    # 计算轮廓面积
    contour_area = cv2.contourArea(contour)

    # 计算最小包围矩形面积
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    box_area = cv2.contourArea(box)

    # 若多边形逼近面积与最小包围矩形面积相差过大，则说明不是矩形
    if math.fabs(contour_area-box_area)/contour_area > 0.05:
        return False
    # 若角度与90度偏差超过一定范围，则说明不是矩形
    for vid in range(0, len(approx)):
        vec_a = approx[vid-1][0] - approx[vid][0]
        vec_b = approx[(vid+1) % len(approx)][0] - approx[vid][0]
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        # 判断当前角度与90度偏差是否超过10度
        cos = np.inner(vec_a, vec_b) / (norm_a*norm_b)
        if cos > math.cos((90-10)*math.pi/180) or cos < math.cos((90+10)*math.pi/180):
            return False

    # 通过矩形判定
    return True


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


def filterContourVertices(contour):
    """去除轮廓contour中重复的顶点。

    Args:
        contour (np.ndarray): 待去除重复顶点的轮廓。

    Returns:
        contour: 去除重复顶点后的轮廓。
    """
    # 删除与已有顶点过近的顶点
    is_valid = np.ones(contour.shape[0], dtype=bool)
    for vid0 in range(0, len(contour)):
        if is_valid[vid0]:
            for vid1 in range(vid0+1, len(contour)):
                vec = contour[vid0] - contour[vid1]
                distance = np.linalg.norm(vec)
                # 若此顶点与已有顶点距离小于5，则可忽略
                if distance < 5:
                    is_valid[vid1] = False
    contour = contour[is_valid, :]
    
    # 删除与相邻顶点夹角过大的顶点
    is_valid = np.ones(contour.shape[0], dtype=bool)
    for vid in range(0, len(contour)):
        vec_a = contour[vid-1] - contour[vid]
        vec_b = contour[(vid+1) % len(contour)] - contour[vid]
        norm_a = np.linalg.norm(vec_a) 
        norm_b = np.linalg.norm(vec_b)
        # 若此顶点与相邻顶点夹角过大（大于160度），则可忽略
        cos = np.inner(vec_a, vec_b) / (norm_a*norm_b)
        if cos < math.cos(math.pi*160/180):
            is_valid[vid] = False
    contour = contour[is_valid, :]

    # 返回筛选后剩下的顶点
    return contour


def polygonDetect(img, denoised, approxs, *polygonContainers):
    """多边形的检测函数。

    Args:
        img (np.ndarray): 原图。
        denoised (np.ndarray): 去噪后的图像。
        approxs (PolygonContainer): 存储所有多边形的容器。
        polygonContainers (list): 存储待检测的类型的多边形的容器列表。
    """
    # 二值化
    thresh = binarize(denoised, "Canny")

    # 轮廓检测
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = np.copy(img)
    cv2.drawContours(img_contours, contours, -1, (0, 0, 0), 2)
    cv2.namedWindow("Contours", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Contours", img_contours)

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

    # 对轮廓进行多边形逼近并分类
    for cid in range(0, len(contours)):
        # 获取当前枚举的轮廓边界顶点与其质心
        c = contours[cid]
        mc = centroids[cid]

        # 去除过小的形状（即噪声）
        if cv2.contourArea(c) < 100:
            continue

        # 用多边形逼近轮廓
        epsilon = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)

        # 过滤掉的无用顶点
        approx = filterContourVertices(approx)

        # 保存轮廓的顶点与质心
        approxs.contours.append(approx)
        approxs.centroids.append(mc)

        # 调试：显示当前多边形，并输出当前多边形的数据
        # showPolygonContours("Approx: %d - vertex_count: %d" % (cid, len(approx)), img, [approx], [mc])
        # print("Approx: %d" % cid)
        # print("Vertex Count: {0}".format(len(approx)))
        # print("Centroid: {0}".format(mc))
        # print("Area: {0}".format(cv2.contourArea(approx)))
        # print("Vertices: {0}".format(approx))
        # print()

        # 判断是否为所需检测的多边形
        vertex_count = len(approx)
        for container in polygonContainers:
            if vertex_count == container.vertex_count and container.check(c, approx):
                container.contours.append(c)
                container.centroids.append(mc)

    # 画出检测到的轮廓的多边形
    showPolygonContours(approxs.name, img, approxs.contours, approxs.centroids)

    # 画出检测到的所需的多边形，并输出个数
    for container in polygonContainers:
        showPolygonContours(container.name, img, container.contours, container.centroids)
        print("{0} Count: {1}".format(container.name, len(container.centroids)))


def circleDetect(img, denoised):
    """圆的检测函数。

    Args:
        img (np.ndarray): 原图。
        denoised (np.ndarray): 去噪后的图像。
    """
    # 用Hough变换进行圆的检测
    circles = cv2.HoughCircles(denoised, cv2.HOUGH_GRADIENT, 1, 30, param1=50, param2=60, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))

    # 对每个圆画出边界与圆心
    img_circles = np.copy(img)
    for i in circles[0, :]:
        # 画圆的边界
        cv2.circle(img_circles, (i[0], i[1]), i[2], (0, 0, 0), 3)
        # 画圆心
        cv2.circle(img_circles, (i[0], i[1]), 2, (0, 0, 0), 5)
    cv2.namedWindow("Circle", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Circle", img_circles)

    # 输出检测到的圆的个数
    print("Circle Count: {0}".format(circles.shape[1]))


def roundRectageDetect(img, denoised):
    """圆角矩形的检测函数。

    Args:
        img (np.ndarray): 原图。
        denoised (np.ndarray): 去噪后的图像。
    """
    # 二值化
    thresh = binarize(denoised, "Canny")

    # 轮廓检测
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = np.copy(img)
    cv2.drawContours(img_contours, contours, -1, (255, 0, 0), 2)
    cv2.namedWindow("Contours", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Contours", img_contours)

    contours_p = []
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])
        if (area > 9 and area < 25):
            contours_p.append(contours[i])
            #x, y, w, h= cv2.boundingRect(contours[i])
            #print("x:", x, "y:", y, "w:", w, "h:", h, "\n")
            #cropped_img = img[y:y+h, x:x+w]
            #cv2.imshow("cropped image", cropped_img);
            print(area)
    img_contours_p = np.copy(img)
    cv2.drawContours(img_contours_p, contours_p, -1, (255, 0, 0), 2)
    cv2.namedWindow("Contours p", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Contours p", img_contours_p)




def shapeDetect(img_path):
    """形状检测函数。

    Args:
        img_path (str): 图像的路径。
    """
    # 从文件中读取图像
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Original", img)

    # 转换为灰度图
    cv2.namedWindow("Gray", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Gray", gray)

    # 去噪
    denoised = denoise(gray, "MedianBlur")

    # 三角形与矩形的检测
    #approxs = PolygonContainer("ApproxPolygons", 0, lambda contour, approx : True)
    #triangles = PolygonContainer("Triangle", 3, lambda contour, approx : True)
    #rectangles = PolygonContainer("Rectangle", 4, rectangleCheck)
    #polygonDetect(img, denoised, approxs, triangles, rectangles)
    #polygonDetect(img, denoised, approxs, rectangles)

    # 圆的检测
    #circleDetect(img, denoised)

    roundRectageDetect(img, denoised)


def main():
    """主函数。
    """
    # 设置文件路径
    img_path = '/home/wqy/rasp_space/videos/50-degree/0001.png'

    # 处理文件路径指向的图像
    shapeDetect(img_path)

    # 按任意键后关闭所有窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

