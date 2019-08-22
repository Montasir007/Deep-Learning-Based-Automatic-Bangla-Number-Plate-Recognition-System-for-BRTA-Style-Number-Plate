#%pylab
import glob
from PIL import ImageFont, ImageDraw, Image
import os.path
import os
from scipy.ndimage import imread
from scipy.ndimage.interpolation import zoom
from scipy.misc import imresize
import pandas as pd
import seaborn as sns
sns.set_style('white')
#%matplotlib inline
import argparse
from skimage.segmentation import clear_border
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from PIL import Image
import scipy.misc
import matplotlib
from matplotlib import pylab, mlab, pyplot
plt = pyplot


#import os
#os.chdir("/home/abrar/darknet/build/darknet/x64")
#path1 = "./darknet detector test data/obj.data yolo-obj.cfg yolo-obj_2000.weights "
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
np_name = args["image"]
#file_name ="bdc.jpg"
#command = path1+file_name
#hi = "./darknet detector test data/obj.data yolo-obj.cfg yolo-obj_2000.weights bdc.jpg"
#os.system(command)

def resize_and_fill(im, X=35, Y = 28):
    i = np.argmin(im.shape[:2])
    i = im.shape[0]/float(im.shape[1]) >= 1.25
    if i == 0:
        test_im = imresize(im, (int(im.shape[0]*float(Y)/im.shape[1]) , Y, 3) )
    if i == 1:
        test_im = imresize(im, (X, int(im.shape[1]*float(X)/im.shape[0]), 3) )

    shape_test = np.array(test_im.shape[:2])
    test_im2 = np.ones((X, Y ,3))
    if i == 0:
        for k in range(3):
            test_im2[:, :, k] = np.concatenate((test_im[:,:,k], np.ones((X-shape_test[0], Y))*np.mean(test_im[-1,:,k])), 0)

    if i == 1:
        for k in range(3):
            test_im2[:, :, k] = np.concatenate((test_im[:,:,k], np.ones((X, Y-shape_test[1]))*np.mean(test_im[:,-1,k])), 1)
    return np.uint8(test_im2)


def resize_and_fill2(im, X=35, Y = 20):
    im = rotate(im, 90)
    i = np.argmin(im.shape[:2])
    i = im.shape[0]/float(im.shape[1]) >= 1.75
    if i == 0:
        test_im = imresize(im, (int(im.shape[0]*float(Y)/im.shape[1]) , Y, 3) )
    if i == 1:
        test_im = imresize(im, (X, int(im.shape[1]*float(X)/im.shape[0]), 3) )

    shape_test = np.array(test_im.shape[:2])
    test_im2 = np.ones((X, Y ,3))
    if i == 0:
        for k in range(3):
            test_im2[:, :, k] = np.concatenate((test_im[:,:,k], np.ones((X-shape_test[0], Y))*np.mean(test_im[-1,:,k])), 0)

    if i == 1:
        for k in range(3):
            test_im2[:, :, k] = np.concatenate((test_im[:,:,k], np.ones((X, Y-shape_test[1]))*np.mean(test_im[:,-1,k])), 1)
    return np.uint8(test_im2)




def imop(name):
    new = Image.open(name)
    width, height = new.size

    a = height/2
    img2 = new.crop((0, 0, width, a))
    img3 = new.crop((0, a, width, height))
    img2.save("im0.jpg")
    img3.save("im1.jpg")
    for j in range(2):
        ref = cv2.imread("im"+str(j)+".jpg")
        a = ref
        a = imutils.resize(a, width=400)

        if(j==0):
            i=0
            k=0
            names=[]
            all_names= []

        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        ref = imutils.resize(ref, width=400)
        ref = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY_INV |
            cv2.THRESH_OTSU)[1]

        # find contours in the MICR image (i.e,. the outlines of the
        # characters) and sort them from left to right
        refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        refCnts = refCnts[0] if imutils.is_cv2() else refCnts[1]
        refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]

        # create a clone of the original image so we can draw on it
        clone = np.dstack([ref.copy()] * 3)

        # loop over the (sorted) contours

        for c in refCnts:
            # compute the bounding box of the contour and draw it on our
            # image
            (x, y, w, h) = cv2.boundingRect(c)
            #cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)


            if(w>30 and h>30 and w<200 and h<200 and h<(2*w)):
                crop_img = a[y:y+h, x:x+w]
                #cv2.imshow("cropped", crop_img)
                #cv2.waitKey(0)

                if(j==1):
                    i = i+1
                    cv2.imwrite("nn_"+str(i)+".jpg", crop_img)
                    st2 = "nn_"+str(i)+".jpg"
                    all_names.append(st2)
                else:
                    k = k+1
                    cv2.imwrite("n_"+str(k)+".jpg", crop_img)
                    string = "n_"+str(k)+".jpg"
                    #print(string)
                    names.append(string)
                    all_names.append(string)
        # show the output of applying the simple contour method
        #cv2.imshow("Simple Method", clone)
        #cv2.waitKey(0)
    return names, all_names


from keras.models import model_from_json
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")

json_file2 = open('dhame.json', 'r')
loaded_model_json2 = json_file2.read()
json_file2.close()
loaded_model2 = model_from_json(loaded_model_json2)
# load weights into new model
loaded_model2.load_weights("dhame.h5")

json_file3 = open('ko.json', 'r')
loaded_model_json3 = json_file3.read()
json_file3.close()
loaded_model3 = model_from_json(loaded_model_json3)
# load weights into new model
loaded_model3.load_weights("ko.h5")
#print("Model Loaded")

name = np_name


names, all_names = imop(name)
#print(all_names)
iimage = Image.open(name)

pilstring =""
pilstring2 = ""

#plt.imshow(iimage)
#plt.show()
from scipy.ndimage import rotate

#print(str(i))
#print("n_"+str(i)+name)
if len(names)!=0:
    img_name = names[0]
    #print(img_name)
    if os.path.isfile(img_name):
        img = cv2.imread(img_name)
        img_cvt = resize_and_fill2(img)
        scipy.misc.imsave('outfile.jpg', img_cvt)
        img = cv2.imread('outfile.jpg')
        img = np.reshape(img,[1,35,20,3])
        a = loaded_model2.predict_classes(img)
        if(a[0]==0):
            #print("ঢাকা-", end = "")
            pilstring = pilstring + "Dhaka-"
            pilstring2 = pilstring2 + "ঢাকা-"
        if(a[0]==1):
            print("মেট্রো-", end = "")



#print("মেট্রো-", end = "")

pilstring = pilstring + "Metro-"
pilstring2 = pilstring2 + "মেট্রো-"


if len(names)!=0:
    img_name = names[-1]
    if os.path.isfile(img_name):
        img = cv2.imread(img_name)
        #img = rotate(img, 90)
        img_cvt = resize_and_fill(img)
        scipy.misc.imsave('outfile.jpg', img_cvt)
        img = cv2.imread('outfile.jpg')
        img = np.reshape(img,[1,35,28,3])
        a = loaded_model3.predict_classes(img)
        if(a[0]==0):
            #print("ক-", end = "")
            pilstring = pilstring + "Ko-"
            pilstring2 = pilstring2 + "ক-"
        if(a[0]==1):
            #print("খ-", end = "")
            pilstring = pilstring + "Kho-"
            pilstring2 = pilstring2 + "খ-"
        if(a[0]==2):
            #print("গ-", end = "")
            pilstring = pilstring + "Go-"
            pilstring2 = pilstring2 + "গ-"
        if(a[0]==3):
            #print("ঘ-", end = "")
            pilstring = pilstring + "Gho-"
            pilstring2 = pilstring2 + "ঘ-"


for i in range(1, 10):

    img_name = "nn_"+str(i)+".jpg"
    if not os.path.isfile(img_name):
    #ignore if no such file is present.
        break
    img = cv2.imread(img_name)
    img_cvt = resize_and_fill(img)
    scipy.misc.imsave('outfile.jpg', img_cvt)
    img = cv2.imread('outfile.jpg')
    img = np.reshape(img,[1,35,28,3])
    a = loaded_model.predict_classes(img)
    b = str(a)

    if(a[0]==0):
        #print("০", end = "")
        pilstring = pilstring+"0"
        pilstring2 = pilstring2 + "০"
    if(a[0]==1):
        #print("১", end = "")
        pilstring = pilstring+"1"
        pilstring2 = pilstring2 + "১"
    if(a[0]==2):
        #print("২", end = "")
        pilstring = pilstring+"2"
        pilstring2 = pilstring2 + "২"
    if(a[0]==3):
        #print("৩", end = "")
        pilstring = pilstring+"3"
        pilstring2 = pilstring2 + "৩"
    if(a[0]==4):
        #print("৪", end = "")
        pilstring = pilstring+"4"
        pilstring2 = pilstring2 + "৪"
    if(a[0]==5):
        #print("৫", end = "")
        pilstring = pilstring+"5"
        pilstring2 = pilstring2 + "৫"
    if(a[0]==6):
        #print("৬", end = "")
        pilstring = pilstring+"6"
        pilstring2 = pilstring2 + "৬"
    if(a[0]==7):
        #print("৭", end = "")
        pilstring = pilstring+"7"
        pilstring2 = pilstring2 + "৭"
    if(a[0]==8):
        #print("৮", end = "")
        pilstring = pilstring+"8"
        pilstring2 = pilstring2 + "৮"
    if(a[0]==9):
        #print("৯", end = "")
        pilstring = pilstring+"9"
        pilstring2 = pilstring2 + "৯"



draw = ImageDraw.Draw(iimage)

width, height = iimage.size
import math

#font = ImageFont.truetype("SolaimanLipi.ttf", math.ceil(width/15.0))
#print(pilstring)
# Draw the text
#draw.text((0, 0), pilstring2, font=font, fill=(0,0,255,128))

# Save the image
cv2_im_processed = cv2.cvtColor(np.array(iimage), cv2.COLOR_RGB2BGR)
#cv2.imwrite("result.png", cv2_im_processed)
plt.imshow(cv2_im_processed)
plt.title(pilstring)
plt.show()

#print(pilstring)

for i in range(1, 15):
    if not os.path.isfile(all_names[i]):
        break
    img_name = all_names[i]
    os.remove(all_names[i])

os.remove("outfile.jpg")
