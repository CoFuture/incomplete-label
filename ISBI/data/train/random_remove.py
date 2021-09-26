import os
import cv2
import math
import glob
import random


def random_remove_rect(img_path, rate=0.25):
    # 读取图片文件，灰度处理
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 根据rate 计算矩形框面积
    image_h = image.shape[0]
    image_w = image.shape[1]
    rect_h = int(math.floor(image_h * math.sqrt(rate)))
    rect_w = int(math.floor(image_w * math.sqrt(rate)))

    # 计算矩形框原点（左上角）取值范围
    h_range = image_h - rect_h
    w_range = image_w - rect_w

    origin_h = random.randint(0, h_range - 1)
    origin_w = random.randint(0, w_range - 1)
    print("origin points", origin_h, origin_w)

    for h in range(rect_h):
        for w in range(rect_w):
            h_new = h + origin_h
            w_new = w + origin_w
            # 原有效信息是黑色的0，擦除后为白色 255
            if image[h_new][w_new] == 0:
                image[h_new][w_new] = 255

    img_removed_path = img_path.replace('label', 'label_in_40')
    # img_removed_path = img_path.split('.')[0] + '_rmv5.png'
    cv2.imwrite(img_removed_path, image)


if __name__ == '__main__':
    images_path = glob.glob('label/*.png')

    for path in images_path:
        random_remove_rect(path, rate=0.4)

    print("Done")
