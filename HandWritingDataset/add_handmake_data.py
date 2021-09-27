import os
import cv2
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import random
dpi = matplotlib.rcParams['figure.dpi']

def get_figure_size(img, dpi):
    height, width, depth = img.shape
    figsize = width / float(dpi), height / float(dpi)
    return figsize

class DataAugment:

    def __init__(self, source_folder, destination_folder):
        self.source_folder = source_folder
        self.destination_folder = destination_folder
        if not os.path.isdir(self.destination_folder):
            os.mkdir(self.destination_folder)


    def resize(self, image):
        image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
        return image

    def erode(self, image):
        """
        腐蝕
        """
        Conv_hsv_Gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
        # 先轉黑底白字
        image[mask == 0] = [0,0,0]
        image[mask == 255] = [255,255,255]

        # 腐蚀范围2x2
        kernel = np.ones((3,3),np.uint8)
        # 迭代次数 iterations=1
        erosion = cv2.erode(image,kernel, iterations=1)
        return erosion, mask

    def dilate(self, image):
        """
        膨脹
        """
        kernel = np.ones((2,2), np.uint8)
        dilation = cv2.dilate(image, kernel, iterations = 1)
        return dilation

    def change_color(self, image, mask):
        """
        改變字的顏色還有背景顏色
        """
        image[mask == 255] = [24,32,78] #轉字顏色
        image[mask == 0] = [255, 248, 230]  #轉成米白色背景
        return image

    def blur(self, image):
        """
        高斯模糊
        """
        image = cv2.GaussianBlur(image, (5, 5), 0)
        return image

    def customer_line(self, image):
        """
        兩橫一豎(case1)
        兩束一橫(case2)
        兩橫(case1)
        一橫一豎(case2)
        一橫(case1)
        一豎(case2)
        """
        line_len = random.randint(1,3)
        case = random.randint(1,2)
        red = np.array([227,23,13])
        row = 128
        column = 128
        if line_len == 3:
            line_position = [random.randint(1,20),random.randint(110,127),random.randint(1,127)]
            print(line_position)
            if case == 1:
                #兩橫一豎
                image[line_position[0],:] = np.tile(red, (128,1))
                image[line_position[1],:] = np.tile(red, (128,1))
                image[:,line_position[2]] = np.tile(red, (128,1))
                
            else:
                #兩豎一橫
                image[:,line_position[0]] = np.tile(red, (128,1))
                image[:,line_position[1]] = np.tile(red, (128,1))
                image[line_position[2],:] = np.tile(red, (128,1))
                
                
        elif line_len == 2:
            line_position = [random.randint(1,20),random.randint(110,127)]
            print(line_position)
            if case == 1:
                #兩橫
                image[line_position[0],:] = np.tile(red, (128,1))
                image[line_position[1],:] = np.tile(red, (128,1))
                
            else:
                #ㄧ豎一橫   
                index = random.randint(0,1)
                image[line_position[index],:] = np.tile(red, (128,1))
                image[:,line_position[1-index]] = np.tile(red, (128,1))
                
        elif line_len == 1:
            line_position = [random.randint(1,20),random.randint(110,127)]
            print(line_position)
            index = random.randint(0,1)
            if case == 1:
                #一橫
                image[line_position[index],:] = np.tile(red, (128,1))
            else:
                #ㄧ豎 
                image[:,line_position[index]] = np.tile(red, (128,1))
        return image

    def save_jpg(self, image, save_name):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_name, image)

    def pipeline(self):
        files = os.listdir(self.source_folder)
        for filename in files:
            img = cv2.imread(os.path.join(self.source_folder,filename))
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = image[8:58,35:85] #原始資料有被補充過了,所以要移除
            image = self.resize(image)
            image,mask = self.erode(image)
            image = self.dilate(image)
            image = self.change_color(image, mask)
            image = self.blur(image)
            image = self.customer_line(image)
            change_filename = filename.replace("png","jpg")
            self.save_jpg(image=image,save_name=os.path.join(self.destination_folder,change_filename))









if __name__ == '__main__':
    source_folder = "/Users/timshieh/Documents/HandWritingCategory/cleaned_data_50_50"
    # img = cv2.imread(folder+'/要_14.png')
    data_augment = DataAugment(source_folder=source_folder, destination_folder='/Users/timshieh/Documents/Reconition_Github/E.SunBankOCR/patch_data2')
    data_augment.pipeline()
    