import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

DEBUG = True
MAZE_FILE_NAME = "../Maze.png"
ROBOT_FILE_NAME = "../Robot.png"
IMAGE_LADYBUG_FILE_NAME = "../Ladybug_small.png"
MAP_FILE_NAME = "../MapBuilt.png"
MAZE_IMAGE_WIDTH_RATIO = 9
MAZE_IMAGE_HEIGHT_RATIO = 5
MAZE_IMAGE_WIDTH = 900
MAZE_IMAGE_HEIGHT = 500


def showImg(img, title):
    """
    Displays an image (img) in a matplotlib plot with the title (title).
    :param img: The image to be displayed
    :param title: The title of the plot
    """
    fig, axes = plt.subplots(figsize=(MAZE_IMAGE_WIDTH_RATIO, MAZE_IMAGE_HEIGHT_RATIO))
    axes.imshow(img)
    axes.set_title(title)
    fig.show()


def filterImg(img, lower=(0, 0, 0), upper=(179, 255, 255)):
    """
    Filters an image in HSV space using the supplied lower and upper thresholds, then converts it to RGB space
    :param img: An image in the HSV color format
    :param lower: An array of three integers representing the lower color threshold in HSV format
    :param upper: An array of three integers representing the upper color threshold in HSV format
    :return: The filtered image, in RGB space
    """
    thresh_lower = np.array(lower)
    thresh_upper = np.array(upper)

    img_mask = cv2.inRange(img, thresh_lower, thresh_upper)

    filtered_img = cv2.bitwise_and(img, img, mask=img_mask)

    return cv2.cvtColor(filtered_img, cv2.COLOR_HSV2RGB)


def getBinaryImg(img, lower_threshold, upper_threshold):
    (_, binary_img) = cv2.threshold(img, lower_threshold, upper_threshold, cv2.THRESH_BINARY)
    return binary_img


def getColorContours(maze_hsv, lower_threshold, upper_threshold):
    maze_filtered = filterImg(maze_hsv, lower=lower_threshold, upper=upper_threshold)
    maze_filtered_gray = cv2.cvtColor(maze_filtered, cv2.COLOR_RGB2GRAY)

    kernel = np.ones((8, 8), np.uint8)
    maze_filtered_gray_closed = cv2.morphologyEx(maze_filtered_gray, cv2.MORPH_CLOSE, kernel)

    filtered_binary = getBinaryImg(maze_filtered_gray_closed, 50, 255)

    if DEBUG:
        showImg(maze_filtered, "Filtered maze")
        showImg(cv2.cvtColor(maze_filtered_gray, cv2.COLOR_GRAY2RGB), "Filtered maze, grayscale")
        showImg(cv2.cvtColor(maze_filtered_gray_closed, cv2.COLOR_GRAY2RGB), "Filtered maze, grayscale, closed")
        showImg(cv2.cvtColor(filtered_binary, cv2.COLOR_GRAY2RGB), "Filtered binary image")

    _, contours, _ = cv2.findContours(filtered_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def getLargestContours(contours, num_contours):
    largestContours = [None] * num_contours



    return tuple(largestContours)


def getCornerstones(maze_hsv):

    magenta_thresh_lower = (140, 200, 180)
    magenta_thresh_upper = (160, 255, 255)

    cyan_thresh_lower = (80, 200, 180)
    cyan_thresh_upper = (100, 255, 255)

    magenta_contours = getColorContours(maze_hsv, magenta_thresh_lower, magenta_thresh_upper)
    cyan_contours = getColorContours(maze_hsv, cyan_thresh_lower, cyan_thresh_upper)

    cornerstone_bounds = []

    if not len(magenta_contours) == 1:
        print(f"Expected 1 magenta circle, found {len(magenta_contours)}")
    else:
        contour_poly = cv2.approxPolyDP(magenta_contours[0], 3, True)
        bounds = cv2.boundingRect(contour_poly)
        cornerstone_bounds.append(bounds)

    if not len(cyan_contours) == 3:
        print(f"Expected 3 cyan circles, found {len(cyan_contours)}")
    else:
        for i, contour in enumerate(cyan_contours):
            contour_poly = cv2.approxPolyDP(contour, 3, True)
            bounds = cv2.boundingRect(contour_poly)
            cornerstone_bounds.append(bounds)

    cornerstone_ellipses = tuple(tuple((bound[0] + bound[2] / 2,
                                        bound[1] + bound[3] / 2,
                                        bound[2] / 2,
                                        bound[3] / 2)) for bound in cornerstone_bounds)

    return tuple(cornerstone_ellipses)


def markCornerstones(maze_rgb, cornerstone_ellipses):
    cyan_rgb = (0, 255, 255)
    magenta_rgb = (255, 0, 255)

    marked_img = maze_rgb.copy()

    for i in range(0, 4):
        # Mark the three cyan cornerstones with magenta circles
        mark_color = cyan_rgb

        # Mark the magenta cornerstone with a cyan circle
        if i > 0:
            mark_color = magenta_rgb

        ellipse = cornerstone_ellipses[i]
        ellipse_center = tuple((int(ellipse[0]), int(ellipse[1])))
        ellipse_axes = tuple((int(ellipse[2]), int(ellipse[3])))

        cv2.ellipse(marked_img,
                    ellipse_center,
                    ellipse_axes,
                    0,
                    0,
                    360,
                    mark_color,
                    3)

    return marked_img


def getOverheadPerspective(maze_img, cornerstone_points):
    transformed_points = np.float32(((MAZE_IMAGE_WIDTH, MAZE_IMAGE_HEIGHT),
                                   (0, 0),
                                  (MAZE_IMAGE_WIDTH, 0),
                                  (0, MAZE_IMAGE_HEIGHT)))

    if DEBUG:
        print(cornerstone_points)
        print(transformed_points)

    H = cv2.getPerspectiveTransform(np.float32(cornerstone_points), transformed_points)

    return cv2.warpPerspective(maze_img, H, (MAZE_IMAGE_WIDTH, MAZE_IMAGE_HEIGHT))


def getWalls(maze_hsv):
    wall_thresh_lower = (10, 85, 228)
    wall_thresh_upper = (25, 120, 255)

    maze_filtered = filterImg(maze_hsv, lower=wall_thresh_lower, upper=wall_thresh_upper)
    maze_filtered_gray = cv2.cvtColor(maze_filtered, cv2.COLOR_RGB2GRAY)

    # kernel = np.ones((8, 8), np.uint8)
    # maze_filtered_gray_closed = cv2.morphologyEx(maze_filtered_gray, cv2.MORPH_CLOSE, kernel)

    filtered_binary = cv2.threshold(maze_filtered_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] #getBinaryImg(maze_filtered_gray_closed, 200, 255)

    # kernel = np.ones((30, 30), np.uint8)
    # filtered_binary_closed = cv2.morphologyEx(filtered_binary, cv2.MORPH_CLOSE, kernel)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(filtered_binary, low_threshold, high_threshold)

    print(edges)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = 0.1  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(maze_hsv) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    lines_edges = cv2.addWeighted(cv2.cvtColor(maze_hsv, cv2.COLOR_HSV2RGB), 0.8, line_image, 1, 0)

    if DEBUG:
        showImg(maze_filtered, "Filtered maze")
        showImg(cv2.cvtColor(maze_filtered_gray, cv2.COLOR_GRAY2RGB), "Filtered maze, grayscale")
        # showImg(cv2.cvtColor(maze_filtered_gray_closed, cv2.COLOR_GRAY2RGB), "Filtered maze, grayscale, closed")
        showImg(cv2.cvtColor(filtered_binary, cv2.COLOR_GRAY2RGB), "Filtered binary image")
        # showImg(cv2.cvtColor(filtered_binary_closed, cv2.COLOR_GRAY2RGB), "Filtered binary image, closed")
        showImg(lines_edges, "Maze with walls marked")

    # _, contours, _ = cv2.findContours(filtered_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # return contours


if __name__ == "__main__":
    maze_bgr = cv2.imread(MAZE_FILE_NAME)
    maze_rgb = cv2.cvtColor(maze_bgr, cv2.COLOR_BGR2RGB)

    showImg(maze_rgb, "Maze in RGB mode")

    maze_hsv = cv2.cvtColor(maze_rgb, cv2.COLOR_RGB2HSV)

    # cornerstone_bounds = getCornerstones(maze_hsv)
    #
    # cornerstone_img = markCornerstones(maze_rgb, cornerstone_bounds)
    #
    # showImg(cornerstone_img, "Maze with cornerstones marked")
    #
    # cornerstone_points = ((bound[0], bound[1]) for bound in cornerstone_bounds)

    getWalls(maze_hsv)

    # maze_overhead = getOverheadPerspective(maze_rgb, cornerstone_points)

    # showImg(cornerstone_img, "Maze from overhead perspective")
