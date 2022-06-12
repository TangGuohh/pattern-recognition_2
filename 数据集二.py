import cv2
import numpy as np
import os
img_dir = "C:\\pythonProject0N\\other_dataset\\"
files = os.listdir(img_dir)   # 读入文件夹
num_png = len(files)       # 统计文件夹中的图片个数
print(files[0])
max_similar_num = 5        # 要显示的最相似的图片个数

def compare_img_hist(img1, img2):
    """
    Compare the similarity of two pictures using histogram(直方图)
        Attention: this is a comparision of similarity, using histogram to calculate

        For example:
         1. img1 and img2 are both 720P .PNG file,
            and if compare with img1, img2 only add a black dot(about 9*9px),
            the result will be 0.999999999953

    :param img1: img1 in MAT format(img1 = cv2.imread(image1))
    :param img2: img2 in MAT format(img2 = cv2.imread(image2))
    :return: the similarity of two pictures
    """
    # Get the histogram data of image 1, then using normalize the picture for better compare
    img1_hist = cv2.calcHist([img1], [1], None, [256], [0, 256])
    img1_hist = cv2.normalize(img1_hist, img1_hist, 0, 1, cv2.NORM_MINMAX, -1)


    img2_hist = cv2.calcHist([img2], [1], None, [256], [0, 256])
    img2_hist = cv2.normalize(img2_hist, img2_hist, 0, 1, cv2.NORM_MINMAX, -1)

    similarity = cv2.compareHist(img1_hist, img2_hist, 0)

    return similarity

def display_all(max_similar_num = 5):
    result_list = []
    for i in files:
        temp_list = []
        for j in files:
            img1_dir = img_dir + str(i)
            img2_dir = img_dir + str(j)
            # print(img1_dir)
            # print(img2_dir)
            img1 = cv2.imread(img1_dir, 1)
            img2 = cv2.imread(img2_dir, 1)
            img1 = cv2.resize(img1, (100,100))
            img2 = cv2.resize(img2, (100,100))
            # print(img1)
            result = compare_img_hist(img1,img2)
            temp_list.append(result)
        result_list.append(temp_list)
    result_list = np.array(result_list)
    print(result_list)
    print(np.argsort(result_list[0])[-10:])
    # dir = img_dir + str(np.argsort(result_list[i])[-10:][0]) + ".jpg"
    # cv2_img = cv2.imread(dir, 1)
    # cv2.imshow("第{}张相似图片".format(j), cv2_img)
    # print("?")
    # cv2.waitKey(0)
    ii = 0
    for i in range(num_png):
        ii += 1
        count = max_similar_num
        arg_index  = np.argsort(result_list[ii])[-(count+1):-1]
        print("第{}张图片相似图片索引：{}".format(ii,arg_index))
        for j in arg_index:
            dir = img_dir + str(files[j])
            print("和第{}张图片第{}相似的是第{}张图片||相似度:{}||路径:{}".format(ii,count,j,result_list[ii][j],dir))
            count -= 1
            cv2_img = cv2.imread(dir,1)
            cv2.imshow("第{}张相似图片".format(j),cv2_img)
            cv2.waitKey(0)


def display(index,max_similar_num):
    result_list = []
    for i in range(num_png):
        n = i + 1
        temp_list = []
        for j in range(num_png):
            m = j + 1
            img1_dir = img_dir + str(n) + ".jpg"
            img2_dir = img_dir + str(m) + ".jpg"
            # print(img1_dir)
            # print(img2_dir)
            img1 = cv2.imread(img1_dir, 1)
            img2 = cv2.imread(img2_dir, 1)

            result = compare_img_hist(img1,img2)
            temp_list.append(result)
        result_list.append(temp_list)
    result_list = np.array(result_list)
    print(np.argsort(result_list[0])[-10:])
    print(np.argsort(result_list[125])[-(max_similar_num+1):-1])
    # dir = img_dir + str(np.argsort(result_list[i])[-10:][0]) + ".jpg"
    # cv2_img = cv2.imread(dir, 1)
    # cv2.imshow("第{}张相似图片".format(j), cv2_img)
    # print("?")
    # cv2.waitKey(0)
    i = index - 1  #第1张照片的index为0
    arg_index  = np.argsort(result_list[i])[-(max_similar_num+1):-1]
    print("第{}张图片相似图片索引：{}".format(index,arg_index))
    for j in arg_index:
        k = j + 1
        dir = img_dir + str(k) + ".jpg"
        print("和第{}张图片第{}相似的是第{}张图片 路径:{}".format(index,max_similar_num,k,dir))
        max_similar_num -= 1
        cv2_img = cv2.imread(dir,1)
        cv2.imshow("第{}张相似图片".format(k),cv2_img)
        cv2.waitKey(0)


if __name__ == '__main__':
    display_all(max_similar_num=3)


