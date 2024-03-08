import rasterio
from rasterio.windows import Window
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def SegmentImgCustom(Img, Mean, Threshold):
    Low_thresh = tuple([v - t for v, t in zip(Mean, Threshold)])
    High_thresh = tuple([v + t for v, t in zip(Mean, Threshold)])
    segmented_image = cv2.inRange(Img, Low_thresh, High_thresh)

    return segmented_image


def CountAndFilter(Segmented_Img, size_x, size_y):
    kernel = np.ones((1, 1), np.uint8)
    Segmented_Image_Filtered = cv2.morphologyEx(Segmented_Img, cv2.MORPH_OPEN, kernel)
    Contours_Filtered, _ = cv2.findContours(Segmented_Image_Filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    num_of_close_to_edge = 0
    close_to_edge = False

    for contours in Contours_Filtered:
        close_to_edge = False
        for point in contours:
            for x, y in point:
                if x < 2 or y < 2 or x > size_x - 3 or y > size_y - 3:
                    num_of_close_to_edge += 1
                    close_to_edge = True
                    break
            if close_to_edge:
                break

    Num_of_pumpkins = len(Contours_Filtered) - num_of_close_to_edge/2

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
        

Mean_CieLAB = (185.51798144,145.62935035,176.06148492)

Total_numofPumpkin = 0


filename = '/home/henrik/Documents/Large_Scale_Drone/Submission 1/Redone/Cut_Pumpkin_Field.tif'
with rasterio.open(filename) as src:

    tif_height, tif_width = src.height, src.width

    chunk_size_x = 900
    chunk_size_y = 400

    height_chunks = math.ceil(tif_height / chunk_size_y)
    width_chunks = math.ceil(tif_width / chunk_size_x)

    for width_chunk in range(width_chunks):
        for height_chunk in range(height_chunks):
            # Define the window boundaries
            window_start_x = width_chunk * chunk_size_x
            window_start_y = height_chunk * chunk_size_y
            window_end_x = min((width_chunk + 1) * chunk_size_x, tif_width)
            window_end_y = min((height_chunk + 1) * chunk_size_y, tif_height)

            window_location = Window(window_start_x, window_start_y, window_end_x - window_start_x, window_end_y - window_start_y)

            img = src.read(window=window_location)

            if img.size > 0:
                temp = img.transpose(1, 2, 0)
                t2 = cv2.split(temp)
                img_cv = cv2.merge([t2[2], t2[1], t2[0]])

                Img_CieLAB = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
                Segmented_CieLAB_Custom = SegmentImgCustom(Img_CieLAB, Mean_CieLAB, (50, 17, 22))

                Num_of_pumpkins_filtered, Segmented_Image_Filtered, Contours_Filtered= CountAndFilter(Segmented_CieLAB_Custom, chunk_size_x, chunk_size_y)
                Total_numofPumpkin = Total_numofPumpkin + Num_of_pumpkins_filtered

            else:
                continue

    print(Total_numofPumpkin)
