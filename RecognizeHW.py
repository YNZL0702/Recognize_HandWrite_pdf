import cv2
import numpy as np
import os
import fitz
import requests
import base64
import time
from tkinter import filedialog
import re


'''''''''
目标区域的识别
'''''''''
def Find_LIMS_No_Contours(imagePath):

    img = cv2.imread(imagePath)
    
    #目标区域位置 
    #roi=img[200:400, 400:850]      
    roi=img[300:400, 410:890]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    sobel_1x = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize = 3)   
    sobel_1y = cv2.Sobel(gray, cv2.CV_8U, 0, 1, ksize = 3)
    combined_sobel=cv2.bitwise_and(sobel_1x, sobel_1y)
    
    blur=cv2.GaussianBlur(combined_sobel,(9,9),0)

    ret, binary = cv2.threshold(blur , 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)


    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

    dilation = cv2.dilate(binary, element2, iterations = 1)

    erosion = cv2.erode(dilation, element1, iterations = 1)

    dilation2 = cv2.dilate(erosion, element2, iterations = 3)

    region = []
    
    contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    for i in range(len(contours)):
        # 遍历所有轮廓
        
        cnt = contours[i]

        area = cv2.contourArea(cnt)
        
        if(area < 4000):
            continue  

      
        rect = cv2.minAreaRect(cnt)
 
        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)

     
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # 筛选那些太细的矩形，留下扁的
        if(height > width * 1.3):
            continue

        region.append(box)

        X_s = [i[0] for i in box]
        Y_s = [i[1] for i in box]
        x1 = min(X_s)
        if x1<0: x1=0
        x2 = max(X_s)
        y1 = min(Y_s)
        if y1<0: y1=0
        y2 = max(Y_s)
        hight = y2 - y1
        width = x2 - x1

        crop_img: object = roi[y1:y1 + hight, x1:x1 + width]

        #* 转为 base64 
        crop_img_encode=cv2.imencode('.jpg',crop_img)[1]
        crop_img_base64=str(base64.b64encode(crop_img_encode))[2:-1]   
        return crop_img_base64


'''''''''
手写文字识别
'''''''''
def Handwriting_recognition (img_base64):      
  
    try:
    
        request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/handwriting"
    
        params = {"image":img_base64}
        access_token = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
        request_url = request_url + "?access_token=" + access_token
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        response = requests.post(request_url, data=params, headers=headers)
        if response:
            #print (response.json())
            orcTxt=response.json()['words_result'][0]['words']
            print(orcTxt)
            return orcTxt
        else:
            print("error")
            return "No LIMS Number Recognized"
            
    except Exception as e:
        print(f"Error occurred: {e}")
        return "No LIMS Number Recognized"



'''''''''
将PDF的每一页转为图片
'''''''''
def PDF_to_JPG(PDF_Path):
    img_paths = []                    

    pdf = fitz.open(PDF_Path) 

    for i,pg in enumerate(range(0, pdf.pageCount)):
        page = pdf[pg]                                                     
        trans = fitz.Matrix(3.0, 3.0).preRotate(0)                          
        pm = page.getPixmap(matrix=trans, alpha=False)                      
        # pm.writePNG(dir_name + os.sep + base_name[:-4] + '_' + '{:0>3d}.png'.format(pg + 1))  
        img_path = PDF_Path[:-4] + '_' + str(pg+1) + '.jpg'
        pm.writePNG(img_path)                                         
        img_paths.append(img_path)
    pdf.close()
    return img_paths


'''''''''
字符串可用来命名文件
'''''''''
def clean_File_Name(file_Name_str):

    inValid_char = r"[\/\\\:\*\?\"\<\>\|]"                   # '/ \ : * ? " < > |'
    return re.sub(inValid_char, "#", file_Name_str)          # 替换为#


'''''''''
同名文件自动重命名
'''''''''
def same_File_Name(text_Name, Name_List):

    # 递归

    _NameList_=Name_List[:]

    if len(_NameList_)==0 or text_Name not in _NameList_:
        return text_Name
   
    text_Name=text_Name[:-4]+'_(1).jpg'
    return same_File_Name(text_Name, _NameList_)



'''''''''
识别后的文字，重命名图片
'''''''''
def ReName_IMG(PDF_Path):                          
      
    _folder_Path=os.path.dirname(PDF_Path)      
    os.chdir(_folder_Path)                       
  
  
    _images_Path=PDF_to_JPG(PDF_Path)             


    _all_file_Names=os.listdir(_folder_Path)      

    for _file_path in _images_Path:

        _file_oldName=os.path.basename(_file_path)

    
        if _file_oldName.endswith('.jpg'):          

            _Lims_Contour=Find_LIMS_No_Contours(_file_path)

            # 延时0.5秒
            time.sleep(0.5)  

            _reg_Txt=Handwriting_recognition(_Lims_Contour)

            _new_Name=clean_File_Name(_reg_Txt)+'.jpg'

            _new_Name=same_File_Name(_new_Name, _all_file_Names)

            os.rename(_file_oldName, _new_Name)
            _all_file_Names.append(_new_Name)




if __name__ == '__main__':
    #pdfPath= ""
    pdfPath = filedialog.askopenfilename( title='选择一个pdf文件', filetypes=[('pdf', '*pdf'), ('All Files', '*')],)
    if pdfPath.endswith('.pdf'):
        ReName_IMG(pdfPath)
    else: 
        print('\n 请选择一个pdf文件')
        exit()
    

