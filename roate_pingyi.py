import os
import numpy as np
import cv2

Angles = [90, 180, 270]
#Trans_Select_Imgs = 160


def read_imgs(imgs_path):
    imgs_name = os.listdir(imgs_path)
    imgs = []
    for img_name in imgs_name:
        img_path = os.path.join(imgs_path, img_name)
        img = cv2.imread(img_path)
        imgs.append(img)
    return imgs


# 旋转
def rotate_img(img, angle):
    (height, width) = img.shape[:2]
    center = (height // 2, width // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1)
    # 旋转图像
    rotate_img = cv2.warpAffine(img, matrix, (width, height))
    return rotate_img


def get_rotate_imgs(imgs, rotate_img_path):
    for idx_img in range(len(imgs)):
        for angle in Angles:
            r_img = rotate_img(imgs[idx_img], angle)
            rotate_img_name = str(idx_img) + str('_') + str(angle) + '.jpg'
            cv2.imwrite(rotate_img_path + rotate_img_name, r_img)


# 平移
def translate_img(img, x_shift, y_shift):
    (height, width) = img.shape[:2]
    # 平移矩阵(浮点数类型)  x_shift +右移 -左移  y_shift -上移 +下移
    matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    # 平移图像
    trans_img = cv2.warpAffine(img, matrix, (width, height))
    return trans_img


def get_trans_imgs(imgs, trans_img_path):
    imgs_num = len(imgs)
    np.random.seed(2)
    random_imgs = np.random.randint(0, imgs_num, Trans_Select_Imgs)
    for img_idx in random_imgs:
        # 获得随机平移坐标
        x_shift = np.random.randint(-8, 8, 1)
        y_shift = np.random.randint(-8, 8, 1)
        trans_img = translate_img(imgs[img_idx], x_shift, y_shift)
        # 保存平移图片和平移坐标
        trans_img_name = str(img_idx) + '_' + str(x_shift[0]) + '_' + str(y_shift[0]) + '.jpg'
        cv2.imwrite(trans_img_path + trans_img_name, trans_img)


if __name__ == "__main__":
    imgs = read_imgs(r"C:\Users\pc\Desktop\jielixiangmu\222\weather\test2\3")
    # 数据增强
    #get_trans_imgs(imgs, r"D:\AAAmc\lv_data\guangdong_round2_train_20181011\classes\jp/")
    get_rotate_imgs(imgs, r"C:\Users\pc\Desktop\jielixiangmu\222\weather\test2\3")