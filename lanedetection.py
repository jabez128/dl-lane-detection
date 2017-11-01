# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from moviepy.editor import VideoFileClip

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
def canny(img, low_threshold, high_threshold):
    #进行Canny变换
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """进行高斯模糊"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    定义ROI区域，其他部分全部变黑
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    """
    将画好hough line的图片和原图片进行叠加
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


def pipeline(img):
    """
    用于处理每一帧图片的pipeline方法
    """
    image = np.array([])
    if type(img) == str:
        image = mpimg.imread(img)
    image = np.array(img)
    grayscale_image = grayscale(image)
    kernel_size = 3
    gaussian_blur_image = gaussian_blur(grayscale_image, kernel_size)
    canny_low_threshold = 50
    canny_high_threshold = 200
    edge_image = canny(gaussian_blur_image, canny_low_threshold, canny_high_threshold)
    image_shape = edge_image.shape
    x_offset=50 # top width is 50 * 2
    y_offset=76
    v1 = (0+x_offset ,image_shape[0] - y_offset)
    v2 = (int(image_shape[1]/2-x_offset), int(image_shape[0]/2+y_offset))
    v3 = (int(image_shape[1]/2+x_offset), int(image_shape[0]/2+y_offset))
    v4 = (image_shape[1]-x_offset,image_shape[0] - y_offset)
    masked_edge_image = region_of_interest(edge_image, np.array([[v1, v2, v3, v4]], dtype=np.int32))
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = (np.pi)/180 # angular resolution in radians of the Hough grid
    threshold = 20      # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 3    # minimum number of pixels making up a line
    max_line_gap = 40   # maximum gap in pixels between connectable line segments
    hough_line_image = hough_lines(masked_edge_image, rho, theta, threshold, min_line_len, max_line_gap)
    sync_image = weighted_img(hough_line_image, image, alpha=0.8, beta=1., gamma=0.)
    return sync_image

def processVideo():
    white_output = './test_videos_output/solidWhiteRight.mp4'
    clip1 = VideoFileClip("./test_videos/solidWhiteRight.mp4").subclip(0,10)
    white_clip = clip1.fl_image(pipeline)
    white_clip.write_videofile(white_output, audio=False)


#pipeline('./test_images/1.jpg')
processVideo()

