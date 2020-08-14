from joblib import dump, load
import mahotas as mt
import cv2
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

chars = ["ả","n","c","h","ấ","t","ủ","a","à","ô","g","ã","b","o","i","ờ","ạ","ự","ỏ","l","ì","m","k","ẻ","ở","v","ố","ể","r","ê","ặ","ẹ","y","u","ộ","ó","q","ý","ơ","á","p","ú","ò","ậ","é","ư","đ","ừ","ứ","ầ","s","ổ","ệ","d","ị","í","â","ớ","ă","ế","x","ù","ọ","ĩ","ử","ồ","ỉ","ợ","ẫ","ỳ","ụ","ẽ","ữ","e","ề"]

def find_contour(main_img):
    gs = cv2.cvtColor(main_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gs, (25,25),0)
    ret_otsu,im_bw_otsu = cv2.threshold(blur,170,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((50,50),np.uint8)
    #closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)
    _, contours, hierarchy = cv2.findContours(im_bw_otsu,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=len)
    contour = contours[len(contours)-1]
    return contour

def extract_features(main_img, vector_size=32):
    gs = cv2.cvtColor(main_img, cv2.COLOR_BGR2GRAY)
    contour = find_contour(main_img)

    x = []
    _,__,w,h = cv2.boundingRect(contour)
    aspect_ratio = float(w)/h
    x.append(aspect_ratio)

    area = cv2.contourArea(contour)
    rectangularity = w*h/area
    x.append(rectangularity)

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area
    x.append(solidity)

    perimeter = cv2.arcLength(contour,True)
    circularity = ((perimeter)**2)/area
    x.append(circularity)

    leftmost = tuple(contour[contour[:,:,0].argmin()][0])
    rightmost = tuple(contour[contour[:,:,0].argmax()][0])
    topmost = tuple(contour[contour[:,:,1].argmin()][0])
    bottommost = tuple(contour[contour[:,:,1].argmax()][0])
    x.append(leftmost[0])
    x.append(rightmost[0])
    x.append(topmost[0])
    x.append(bottommost[0])
    x.append(leftmost[1])
    x.append(rightmost[1])
    x.append(topmost[1])
    x.append(bottommost[1])

    equi_diameter = np.sqrt(4*area/np.pi)
    (_,y),(MA,ma),angle = cv2.fitEllipse(contour)
    x.append(_)
    x.append(y)
    x.append(MA)
    x.append(ma)
    x.append(angle)
    
    textures = mt.features.haralick(gs)
    ht_mean = textures.mean(axis=0)
    for i in [1,2,4,8]:
      x.append(ht_mean[i])
    return np.array(x)

def Predict(img):
    instance = extract_features(img)
    instance = np.array(instance)
    instance = instance.reshape(1,instance.shape[0])
    DTC_clf = load('weight/DTC_3k.joblib') 
    prediction = DTC_clf.predict(instance)
    print(chars[prediction[0]])
    return chars[prediction[0]]