import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import random as rng

def compare_original_and_segmented_image(original, segmented, title):
    plt.figure(figsize=(9, 3))
    ax1 = plt.subplot(1, 2, 1)
    plt.title(title)
    ax1.imshow(original)
    ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
    ax2.imshow(segmented)

def FindMeanAndAverage(Img, Mask):
    Lower_limit = (0, 0, 253)
    Upper_limit = (0, 0, 255)
    Mask = cv2.inRange(Mask, Lower_limit, Upper_limit)  
    Mean_RGB, Std_RBG = cv2.meanStdDev(Img, mask = Mask)

    return Mean_RGB, Std_RBG

def SegmentImgCustom(Img, Mean, Threshold):
    Values = [value[0] for value in Mean]
    Low_thresh = tuple([v - t for v, t in zip(Values, Threshold)])
    High_thresh = tuple([v + t for v, t in zip(Values, Threshold)])
    segmented_image = cv2.inRange(Img, Low_thresh, High_thresh)

    return segmented_image

def CountPumpkins(Segmented_Img):
    Contours, _ = cv2.findContours(Segmented_Img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    Num_of_pumpkins = len(Contours)

    return Num_of_pumpkins, Contours

def CountAndFilter(Segmented_Img):
    kernel = np.ones((5, 5), np.uint8)
    Segmented_Image_Filtered = cv2.morphologyEx(Segmented_Img, cv2.MORPH_OPEN, kernel)
    Contours_Filtered, _ = cv2.findContours(Segmented_Image_Filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    Num_of_pumpkins = len(Contours_Filtered)

    return Num_of_pumpkins, Segmented_Image_Filtered, Contours_Filtered

def ShowPumpkins(Countours, Original_img):
    contours_poly = [None]*len(Countours)
    boundRect = [None]*len(Countours)
    centers = [None]*len(Countours)
    radius = [None]*len(Countours)
    for i, c in enumerate(Countours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

    drawing = Original_img.copy()
    
    cv2.drawContours(drawing, Countours, -1, (255, 20, 20), 3)

    for i in range(len(Countours)):
        #color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        color = (255, 20, 20)
        # cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
        

    return drawing

Img_path = '/home/henrik/Documents/Large_Scale_Drone/Submission 1/Redone/Snippet.JPG'
Img = cv2.imread(Img_path)

Annotated_img = '/home/henrik/Documents/Large_Scale_Drone/Submission 1/Redone/Snippet_Annotated.JPG'
Img_Annotated = cv2.imread(Annotated_img)

Img_CieLAB = cv2.cvtColor(Img, cv2.COLOR_BGR2LAB)

Mean_CieLAB, Std_CieLAB = FindMeanAndAverage(Img_CieLAB, Img_Annotated)

print(Mean_CieLAB)

Segmented_CieLAB_Custom = SegmentImgCustom(Img_CieLAB, Mean_CieLAB, (50, 17, 22))

Num_of_pumpkins, _ = CountPumpkins(Segmented_CieLAB_Custom)
print("Number of pumpkins in unfiltered image: ",Num_of_pumpkins)

Num_of_pumpkins_filtered, Segmented_Image_Filtered, Contours_Filtered = CountAndFilter(Segmented_CieLAB_Custom)
print("Number of pumpkins in filtered image: ", Num_of_pumpkins_filtered)
cv2.imwrite("Output/Filtered_Segmented_Image.JPG", Segmented_Image_Filtered)

Pumpkins_Visualised = ShowPumpkins(Contours_Filtered, Img)
cv2.imwrite("Output/Pumpkins_Visualised.JPG", Pumpkins_Visualised)
