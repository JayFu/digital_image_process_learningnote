import cv2
import numpy as np

the_img = cv2.imread("lena.ppm")
gray_img = cv2.imread("lena.ppm", cv2.IMREAD_GRAYSCALE)
cv2.namedWindow("OriginImage")
cv2.imshow("OriginImage", the_img)
cv2.imshow("OriginImage", gray_img)
cv2.waitKey(1000)
print("start!")

# 输出图像
def print_img(img, name):
    cv2.namedWindow(name)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    print("finished!")

# 降低图像灰度级数
def Grayscale_modify(img, target_scale):
    print("start modification")
    size = img.shape
    scale = 2**8 // 2**target_scale
    scale_matrix = np.full(size, scale, dtype=int)
    # print(img)
    dst_img = img // scale_matrix * scale_matrix
    dst_img = dst_img.astype(np.int8)
    # print(dst_img)
    print_img(dst_img, "grayscale decrease")

# Grayscale_modify(gray_img, 3)

# 图像内插
# 最近邻内插法

# 双线性内插


# 欧式距离

# 街区距离

# 棋盘距离

# 图像平均

# 图像相减

# 图像相乘除

# 仿射变换

# 图像配准

# // 傅里叶变换