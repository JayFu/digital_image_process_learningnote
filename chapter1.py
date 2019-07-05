import cv2
import numpy as np

the_img = cv2.imread("lena.ppm")
gray_img = cv2.imread("lena.ppm", cv2.IMREAD_GRAYSCALE)
cv2.namedWindow("OriginImage")
cv2.imshow("OriginImage", the_img)
cv2.imshow("OriginImage", gray_img)
cv2.waitKey(1000)
print("ori", gray_img)
print("start!")


# 输出图像
def print_img(img, name):
    cv2.namedWindow(name)
    cv2.imshow(name, img)
    cv2.waitKey(5000)
    print("finished!")

# 降低图像灰度级数
def Grayscale_modify(img, target_scale):
    print("start modification")
    size = img.shape
    scale = 2**8 // 2**target_scale
    scale_matrix = np.full(size, scale, dtype=np.uint8)
    # print(img)
    dst_img = img // scale_matrix * scale_matrix
    dst_img = dst_img.astype(np.int8)
    # print(dst_img)
    print_img(dst_img, "grayscale decrease")

# Grayscale_modify(gray_img, 3)

# 图像内插
# 最近邻内插法
def nearest_neighbor_interpolation(img, times):
    print("start interpolation")
    size = img.shape
    dst_size = (int(size[0] * times), int(size[1] * times))
    dst = np.zeros(dst_size, dtype=np.uint8)
    for i in range(dst_size[0]):
        for j in range(dst_size[1]):
            if (i/times) - (i / times // 1) > 0.5:
                dst_x = int(i / times // 1) + 1
            else:
                dst_x = int(i / times // 1)
            if (j / times) - (j / times // 1) > 0.5:
                dst_y = int(j / times // 1) + 1
            else:
                dst_y = int(j / times // 1)
            
            dst[i][j] = img[dst_x][dst_y]
            if dst[i, j] < 0:
                print(i, j, dst_x, dst_y)
                exit()
    print(dst[0][0], dst[0][0] > 0)
    print_img(dst, "nearest neighbor interpolation")
    print("nni", dst)
# nearest_neighbor_interpolation(gray_img, 1.5)

# 双线性内插
def bilinear_interpolation(img, times):
    print("start interpolation")
    size = img.shape
    dst_size = (int(size[0] * times), int(size[1] * times))
    dst = np.zeros(dst_size, dtype=np.uint8)
    for i in range(dst_size[0]):
        for j in range(dst_size[1]):
            ori_x = i / times
            ori_y = j / times
            if ori_x > size[0] - 1:
                ori_x -= 1
            if ori_y > size[1] - 1:
                ori_y -= 1
            neighbors = np.zeros((2,2), dtype=np.int)
            neighbors[0][0], neighbors[0][1], neighbors[1][0], \
                neighbors[1][1] = int(img[int(ori_x)][int(ori_y)]), \
                int(img[int(ori_x)][int(ori_y)+1]), int(img[int(ori_x)+1][int(ori_y)]), \
                int(img[int(ori_x)+1][int(ori_y)+1]) 
            r1 = abs(neighbors[1][0] - neighbors[0][0]) * (ori_x - int(ori_x)) + neighbors[0][0]
            r2 = abs(neighbors[1][1] - neighbors[1][0]) * (ori_x - int(ori_x)) + neighbors[1][0]
            dst_value = (r2 - r1) * (ori_y - int(ori_y)) + r1
            dst[i][j] = dst_value
    print("bili", dst)
    print(np.shape(dst))
    print_img(dst, "bilinear_interpolation")
# bilinear_interpolation(gray_img, 1.5)

# 图像平均
def add_salt_noise(img, percentage):
    row, column = img.shape
    noise_salt = np.random.randint(0, 256, (row, column))
    noise_salt = np.where(noise_salt < percentage * 256, 255, 0)
    img.astype("float")
    noise_salt.astype("float")
    salt = img + noise_salt
    salt = np.where(salt > 255, 255, salt)
    salt.astype("uint8")
    print_img(img, "noised")
    return img

def img_average_denoise(img):
    salt_noise_percentage = 0.04
    img = add_salt_noise(img, salt_noise_percentage)
    img.astype(np.uint32)
    img_shape = img.shape
    for i in range(1, img_shape[0] - 1):
        for j in range(1, img_shape[1] - 1):
            counter = 0
            sum = img[i, j]
            for x in [-1, 0, 1]:
                for y in [-1, 0, 1]:
                    if i + x >= 0 & j + y >=0:
                        sum += img[i+x, j+y]
                        counter += 1
            img[i, j] = img[i, j] // counter
    
    img.astype(np.uint8)
    print_img(img, "average")

img_average_denoise(gray_img)
# 图像相减

# 图像相乘除

# 仿射变换

# 图像配准

# 傅里叶变换