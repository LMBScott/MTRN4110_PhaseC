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


def getCornerstones(maze_img):
    maze_hsv = cv2.cvtColor(maze_img, cv2.COLOR_RGB2HSV)

    cyan_thresh_lower = (89, 230, 230)
    cyan_thresh_upper = (90, 255, 255)

    maze_cyan = filterImg(maze_hsv, lower=cyan_thresh_lower, upper=cyan_thresh_upper)
    maze_cyan_gray = cv2.cvtColor(maze_cyan, cv2.COLOR_RGB2GRAY)

    magenta_thresh_lower = (140, 230, 230)
    magenta_thresh_upper = (160, 255, 255)

    maze_magenta = filterImg(maze_hsv, lower=magenta_thresh_lower, upper=magenta_thresh_upper)
    maze_magenta_gray = cv2.cvtColor(maze_magenta, cv2.COLOR_RGB2GRAY)

    kernel = np.ones((8, 8), np.uint8)

    circle_accumulation = 0.8
    circle_min_dist = 50
    circle_param_1 = 5
    circle_param_2 = 10
    circle_min_radius = 3
    circle_max_radius = 15

    cornerstone_points = []
    cornerstone_radii = []

    maze_magenta_gray_closed = cv2.morphologyEx(maze_magenta_gray, cv2.MORPH_CLOSE, kernel)

    magenta_circles = cv2.HoughCircles(maze_magenta_gray_closed,
                                       cv2.HOUGH_GRADIENT,
                                       circle_accumulation,
                                       circle_min_dist,
                                       param1=circle_param_1,
                                       param2=circle_param_2,
                                       minRadius=circle_min_radius,
                                       maxRadius=circle_max_radius)

    if magenta_circles is None:
        print("Couldn't find any magenta circles")
    elif not len(magenta_circles[0]) == 1:
        print(f"Expected 1 magenta circle, found {len(magenta_circles)}")
    else:
        cornerstone_points.append((magenta_circles[0][0][0], magenta_circles[0][0][1]))
        cornerstone_radii.append(magenta_circles[0][0][2])

    maze_cyan_gray_closed = cv2.morphologyEx(maze_cyan_gray, cv2.MORPH_CLOSE, kernel)

    cyan_circles = cv2.HoughCircles(maze_cyan_gray_closed,
                                    cv2.HOUGH_GRADIENT,
                                    circle_accumulation,
                                    circle_min_dist,
                                    param1=circle_param_1,
                                    param2=circle_param_2,
                                    minRadius=circle_min_radius,
                                    maxRadius=circle_max_radius)

    if cyan_circles is None:
        print("Couldn't find any cyan circles")
    elif not len(cyan_circles[0]) == 3:
        print(f"Expected 3 cyan circles, found {len(cyan_circles)}")
    else:
        for circle in cyan_circles[0]:
            cornerstone_points.append((circle[0], circle[1]))
            cornerstone_radii.append(circle[2])

    if DEBUG:
        showImg(maze_cyan, "Cyan filtered maze")
        showImg(cv2.cvtColor(maze_cyan_gray, cv2.COLOR_GRAY2RGB), "Cyan filtered maze, grayscale")
        showImg(cv2.cvtColor(maze_cyan_gray_closed, cv2.COLOR_GRAY2RGB), "Cyan filtered maze, grayscale, closed")
        showImg(maze_magenta, "Magenta filtered maze")
        showImg(cv2.cvtColor(maze_magenta_gray, cv2.COLOR_GRAY2RGB), "Magenta filtered maze, grayscale")
        showImg(cv2.cvtColor(maze_magenta_gray_closed, cv2.COLOR_GRAY2RGB), "Magenta filtered maze, grayscale, closed")

    return (cornerstone_points, cornerstone_radii)


def markCornerstones(maze_rgb, cornerstone_points, cornerstone_radii):
    cyan_rgb = (0, 255, 255)
    magenta_rgb = (255, 0, 255)

    marked_img = maze_rgb.copy()

    for i in range(0, 4):
        # Mark the three cyan cornerstones with magenta circles
        mark_color = cyan_rgb

        # Mark the magenta cornerstone with a cyan circle
        if i > 0:
            mark_color = magenta_rgb

        point = cornerstone_points[i]

        cv2.circle(marked_img, (point[0], point[1]), cornerstone_radii[i], mark_color, 3)

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


if __name__ == "__main__":
    maze_bgr = cv2.imread(MAZE_FILE_NAME)
    maze_rgb = cv2.cvtColor(maze_bgr, cv2.COLOR_BGR2RGB)

    showImg(maze_rgb, "Maze in RGB mode")

    (cornerstone_points, cornerstone_radii) = getCornerstones(maze_rgb)

    cornerstone_img = markCornerstones(maze_rgb, cornerstone_points, cornerstone_radii)

    showImg(cornerstone_img, "Maze with cornerstones marked")

    # maze_overhead = getOverheadPerspective(maze_rgb, cornerstone_points)

    # showImg(cornerstone_img, "Maze from overhead perspective")
