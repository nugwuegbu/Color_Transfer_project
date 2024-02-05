import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import imutils
import pprint
from matplotlib import pyplot as plt


def createSkin(image):

    img = image.copy()
    # cv2.imshow("main",img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    #HSV thresholds

    lower = np.array([0,48,80],dtype=np.uint8)
    upper = np.array([20,255,255],dtype=np.uint8)

    skinMask = cv2.inRange(img,lower,upper)

    #cleaning skinMask using Gaussian Filter
    skinMask = cv2.GaussianBlur(skinMask,(3,3),0)

    #Sking extraction from the threshold
    skin = cv2.bitwise_and(img,img,mask=skinMask)

    #return skin image
    return cv2.cvtColor(skin,cv2.COLOR_HSV2BGR)

def remove_black(estimator_labels,estimator_cluster):
    #check black
    hasBlack = False

    #Get number of occurrence for each color
    occurence_counter = Counter(estimator_labels)

    #Quick lambda function to compare to lists
    def compare(x,y):return Counter(x) == Counter(y)

    # Loop through the most common occuring colo
    for x in occurence_counter.most_common(len(estimator_cluster)):

        #List comprehension to convert each RBG number to int
        color = [int(i) for i in estimator_cluster[x[0]].tolist()]

        # Check if the color is [0,0,0] that if it is black
        if compare(color,[0,0,0]):
            #delete the occurrence
            del occurence_counter[x[0]]
            #remove the cluster
            hasBlack = True
            estimator_cluster = np.delete(estimator_cluster,x[0],0)
            break

    return (occurence_counter,estimator_cluster,hasBlack)



def get_color_info(estimator_labels, estimator_cluster, hasThresholding=False):
    #variable to keep count of occurance of each color
    occurance_counter = None

    #Output list variable to return
    colorInformation = []

    #check for black
    hasBlack = False

    #if mask has been applied remove the black
    if hasThresholding == True:

        (occurance,cluster,black) = remove_black(estimator_labels,estimator_cluster)
        occurance_counter = occurance
        estimator_cluster = cluster
        hasBlack = black
    else:
        occurance_counter = Counter(estimator_labels)

    #get the total sum of all the predicted occurances
    totalOccurance = sum(occurance_counter.values())

    #loop through all the predicted colors
    for x in occurance_counter.most_common(len(estimator_cluster)):
        index = (int(x[0]))

        #Quick fix for index out of bound when there's no threshold
        index = (index-1) if(hasThresholding & hasBlack) & (int(index) != 0) else index

        #Get the color number into the list
        color = estimator_cluster[index].tolist()

        #Get the percentage of each color
        color_percentage = (x[1]/totalOccurance)

        #make the dictionary of the information
        colorInfo = {"cluster_index":index,"color":color,"color_percentage":color_percentage}

        #Add the dictionary to the list
        colorInformation.append(colorInfo)
    # print("color in array",colorInformation)
    return colorInformation


def extract_dominant_color(image,number_of_colors=5,hasThresholding=False):
    #Quick fix Increase cluster counter to neglect the black
    if hasThresholding == True:
        number_of_colors += 1

    img = image.copy()
    # cv2.imshow('reshaped image', img)

    #convert image into RBG color space
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    #reshape image
    img = img.reshape((img.shape[0]*img.shape[1]),3)


    #initiate Kmeans object
    estimator = KMeans(n_clusters=number_of_colors,random_state=0)

    #Fit the image
    estimator.fit(img)

    #Get color Information
    colorInformation = get_color_info(estimator.labels_,estimator.cluster_centers_,hasThresholding)
    # print(colorInformation)
    return colorInformation

def plot_color(colorInfo):
    #Create a 500 * 500 black image
    color_bar = np.zeros((100,500,3),dtype="uint8")

    top_x = 0
    for x in colorInfo:
        bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1])

        color = tuple(map(int,(x['color'])))

        cv2.rectangle(color_bar,(int(top_x),0),(int(bottom_x),color_bar.shape[0]),color,-1)
        top_x = bottom_x
        return color_bar

