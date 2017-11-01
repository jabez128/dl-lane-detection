# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from moviepy.editor import VideoFileClip

import math

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=6):
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2 - y1)/(x2 - x1)
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            top_horizon_line = ([0, img.shape[0]*0.5], [img.shape[1], img.shape[0]*0.7])
            bottom_horizon_line = ([0, img.shape[0]], [img.shape[1], img.shape[0]])
            line_intersection_top = line_intersection(top_horizon_line, ([x1, y1], [x2, y2]))
            line_intersection_bottom = line_intersection(bottom_horizon_line, ([x1, y1], [x2, y2]))
            if line_intersection_top == None or line_intersection_bottom == None:
                return
            cv2.line(img, line_intersection_top, line_intersection_bottom, color, thickness)


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x), int(y)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    """
    initial_img * α + img * β + λ
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


def pipeline(img):
    """
    用于处理每一帧图片的pipeline方法
    """
    image = np.array([])
    showImg = False
    if type(img) == str:
        image = mpimg.imread(img)
        showImg = True
    else:
        image = np.array(img)
    image = np.array(image)
    grayscale_image = grayscale(image)
    kernel_size = 3
    gaussian_blur_image = gaussian_blur(grayscale_image, kernel_size)
    canny_low_threshold = 50
    canny_high_threshold = 200
    edge_image = canny(gaussian_blur_image, canny_low_threshold, canny_high_threshold)
    image_shape = edge_image.shape
    x_offset=50
    y_offset=76
    v1 = (0+x_offset ,image_shape[0] - y_offset)
    v2 = (int(image_shape[1]/2-x_offset), int(image_shape[0]/2+y_offset))
    v3 = (int(image_shape[1]/2+x_offset), int(image_shape[0]/2+y_offset))
    v4 = (image_shape[1]-x_offset,image_shape[0] - y_offset)
    masked_edge_image = region_of_interest(edge_image, np.array([[v1, v2, v3, v4]], dtype=np.int32))
    rho = 2
    theta = (np.pi)/180
    threshold = 20
    min_line_len = 3
    max_line_gap = 40
    hough_line_image = hough_lines(masked_edge_image, rho, theta, threshold, min_line_len, max_line_gap)
    sync_image = weighted_img(hough_line_image, image, alpha=0.8, beta=1., gamma=0.)
    if showImg:
        plt.imshow(sync_image)
        plt.show()
    return sync_image

def processVideo():
    white_output = './test_videos_output/solidWhiteRight.mp4'
    clip1 = VideoFileClip("./test_videos/solidWhiteRight.mp4").subclip(0,10)
    white_clip = clip1.fl_image(pipeline)
    white_clip.write_videofile(white_output, audio=False)


#pipeline('./test_images/2.jpg')
processVideo()

