import os
import pickle

import cv2
import numpy as np
from PIL import Image

label_dir = r'M:\Users\jack\Desktop\C4\CK+DB\Emotion_labels'
image_dir = r'M:\Users\jack\Desktop\C4\CK+DB\cohn-kanade-images'

# 期望图片大小
purpose_size = 120
face_cascade = cv2.CascadeClassifier('../resources/haarcascade_frontalface_default.xml')


# 裁剪人脸部分
def image_cut(file_name):
    # cv2读取图片
    im = cv2.imread(file_name)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # cv2检测人脸中心区域
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(5, 5)
    )

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # PIL读取图片
            img = Image.open(file_name)
            # 转换为灰度图片
            img = img.convert("L")
            # 裁剪人脸核心部分
            crop = img.crop((x, y, x + w, y + h))
            # 缩小为120*120
            crop = crop.resize((purpose_size, purpose_size))
            return crop
    return None


# 将图片转换为数据矩阵
def image_to_matrix(filename):
    # 裁剪并缩小
    img = image_cut(filename)
    data = img.getdata()
    # 归一化到(0,1)
    return np.array(data, dtype=float) / 255.0


# 获取文件中的label值
def get_label(file_name):
    f = open(file_name, 'r+')
    line = f.readline()  # only one row
    line_data = line.split(' ')
    label = float(line_data[3])
    f.close()
    # 1-7 的标签值转为 0-6
    return int(label) - 1


# 保存脸部核心区域的数据到pickle文件中
def save_picture_data():
    # like [[data1, label1], [data2, label2], ...]
    data_label = []

    # 获取子目录列表, like ['S005\001', 'S010\001', 'S010\002', ...]
    dir_list = []
    for root, dirs, _ in os.walk(image_dir):
        for rdir in dirs:
            for _, sub_dirs, _ in os.walk(root + '\\' + rdir):
                for sub_dir in sub_dirs:
                    dir_list.append(rdir + '\\' + sub_dir)
                break
        break

    # 遍历目录获取文件
    for path in dir_list:
        # 处理 images
        for root, _, files in os.walk(image_dir + '\\' + path):
            for i in range(0, len(files)):
                if files[i].split('.')[1] == 'png':
                    # 裁剪图片，并将其转为数据矩阵
                    img_data = image_to_matrix(root + '\\' + files[i])
                    # 处理相应的 label
                    for lroot, _, lfiles in os.walk(label_dir + '\\' + path):
                        if len(lfiles) > 0:  # picture has label
                            label = get_label(lroot + '\\' + lfiles[0])
                            data_label.append([img_data, label])
                        break
            break
    # 写入数据到pkl文件
    pkl_file = '../data/data_label_list_120.pkl'
    with open(pkl_file, 'wb') as f:
        pickle.dump(data_label, f)
        f.close()
    print('Picture Data Saved Successfully into:', pkl_file)


if __name__ == '__main__':
    save_picture_data()
    print('\n--------------------------Program Finished---------------------------\n')
