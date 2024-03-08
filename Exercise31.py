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


def PlotColorHistogram(image, title):
    colors = ('b', 'g', 'r')
    plt.figure(figsize=(8, 6))
    for i, color in enumerate(colors):
        histogram = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(histogram, color=color)
        plt.xlim([0, 256])
    plt.title(title)
    plt.xlabel('Pixel value')
    plt.ylabel('Frequency')
    plt.show()

def SegmentImg(Img, Mean, Threshold):
    Values = [value[0] for value in Mean]
    Values_T = [value[0] for value in Threshold]
    Low_thresh = tuple([v - t for v, t in zip(Values, Values_T)])
    High_thresh = tuple([v + t for v, t in zip(Values, Values_T)])
    segmented_image = cv2.inRange(Img, Low_thresh, High_thresh)

    return segmented_image

def SegmentImgCustom(Img, Mean, Threshold):
    Values = [value[0] for value in Mean]
    Low_thresh = tuple([v - t for v, t in zip(Values, Threshold)])
    High_thresh = tuple([v + t for v, t in zip(Values, Threshold)])
    segmented_image = cv2.inRange(Img, Low_thresh, High_thresh)

    return segmented_image

def euclidean_segmentation(image, Mean, threshold):
    reference_color = [value[0] for value in Mean]
    euclidean_distances = np.linalg.norm(image - reference_color, axis=-1)
    segmented = (euclidean_distances < threshold).astype(np.uint8) * 255
    return segmented


Img_path = '/home/henrik/Documents/Large_Scale_Drone/Submission 1/Redone/Snippet.JPG'
Img = cv2.imread(Img_path)

Annotated_img = '/home/henrik/Documents/Large_Scale_Drone/Submission 1/Redone/Snippet_Annotated.JPG'
Img_Annotated = cv2.imread(Annotated_img)

Img_CieLAB = cv2.cvtColor(Img, cv2.COLOR_BGR2LAB)


Mean_RGB, Std_RGB = FindMeanAndAverage(Img, Img_Annotated)
print("Mean Value RBG: \n", Mean_RGB, "\n Standard deviation RBG: \n", Std_RGB)

Mean_CieLAB, Std_CieLAB = FindMeanAndAverage(Img_CieLAB, Img_Annotated)
print("Mean Value CieLAB: \n", Mean_CieLAB, "\n Standard deviation CieLAB: \n", Std_CieLAB)

#Uncomment to see Histograms
#PlotColorHistogram(Img, "RGB Histogram")
#PlotColorHistogram(Img_CieLAB, "CieLAB Histogram")

Segmented_RBG_Std = SegmentImg(Img, Mean_RGB, Std_RGB)
cv2.imwrite("Output/Segmented_RBG_Std.JPG", Segmented_RBG_Std)
Segmented_CieLAB_Std = SegmentImg(Img_CieLAB, Mean_CieLAB, Std_CieLAB)
cv2.imwrite("Output/Segmented_CieLAB_Std.JPG", Segmented_CieLAB_Std)

Segmented_RBG_Custom = SegmentImgCustom(Img, Mean_RGB, (40, 40, 70))
cv2.imwrite("Output/Segmented_RBG_custom.JPG", Segmented_RBG_Custom)
Segmented_CieLAB_Custom = SegmentImgCustom(Img_CieLAB, Mean_CieLAB, (50, 17, 22))
cv2.imwrite("Output/Segmented_CieLAB_custom.JPG", Segmented_CieLAB_Custom)


euclidean_threshold = 65
segmented_euclidean = euclidean_segmentation(Img, Mean_RGB, euclidean_threshold)
cv2.imwrite("Output/Segmented_Euclidean.JPG", segmented_euclidean)
