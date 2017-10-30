"""
Test the trained model
"""
import pickle
import numpy as np
import os
import cv2
from sklearn.neural_network import MLPClassifier


#load saved model for prediction
with open('trained_model.pickle', 'rb') as f:
    clf = pickle.load(f)

with open('trained_objects.pickle','rb') as ff:
    gest_identifier=pickle.load(ff)

print(gest_identifier)


def extraction(frame):
    #convert from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #define range of green color in HSV
    lower_green = np.array([70,60,60])
    upper_green = np.array([90,255,255])
    #Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_green, upper_green)
    #Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    #convert to grayscale 
    bgr = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    #find connected components (white blobs)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(gray, connectivity = 4)
    #connectedComponentswithStats gives information on every blob e.g size
    #taking out the background which is also considered a component
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    # minimum size of particles we want to keep (number of pixels)
    min_size = 1500
    #resulting image
    blob = np.zeros((output.shape))
    #for every blob in the image, remove blobs smaller than set size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            blob[output == i + 1] = 255
    #Coordinates of non-black pixels.
    coords = np.argwhere(blob)
    #Bounding box of non-black pixels.
    try:
        x0, y0 = coords.min(axis=0) 
        x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top
        #Get the contents of the bounding box.
        cropped = gray[x0:x1, y0:y1]
        #scale back to original size to ensure all frames are equal size
        height, width = gray.shape[:2]
        new = cv2.resize(cropped,(width//2, height//2), interpolation = cv2.INTER_CUBIC)
        #blur image to smooth edges
        blur = cv2.bilateralFilter(new,15,75,75)
        return blur
    except:
        pass

def load_data():
    x_data=[]
    path=os.path.dirname(os.path.abspath(__file__))
    for testframe in os.listdir(path+"\\testdata"):    #for each gesture folder
        try:    #try read the .jpg files, most others should throw an exception
            img = cv2.imread(path+"\\testdata\\"+testframe)
            #print(path+"\\data\\frames\\"+gesture+"\\"+framefile)
            if img is not None:
                nimg = extraction(img)
                #print("hi")
                try:
                    x_data.append(nimg.flatten())   
                except:
                    print(testframe)
                    continue
        except:
            continue
    x_data=np.array(x_data)
    #print(np.shape(x_data))
    return x_data

x_data=load_data()

probs = clf.predict_proba(x_data)
print(probs)

