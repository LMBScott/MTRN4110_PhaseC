import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

DEBUG = False
MAZE_FILE_NAME = "../Maze_2.png"
ROBOT_FILE_NAME = "../Robot_2.png"
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
    """
    Get the (num_contours) largest contours in a list of contours
    :param contours: The list of contours
    :param num_contours: The number of largest contours to find
    :return: A tuple of the (num_contours) largest contours
    """
    largest_contours = [None] * num_contours
    largest_contour_areas = [0] * num_contours

    # Index of the smallest contour in the list of largest contours
    min_contour_index = 0
    min_contour_area = 0

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])

        if area > min_contour_area:
            largest_contours[min_contour_index] = contours[i]
            largest_contour_areas[min_contour_index] = area
            min_contour_area = area

            # Search through largest contour areas and find the new min contour area
            for j in range(len(largest_contour_areas)):
                if largest_contour_areas[j] < min_contour_area:
                    min_contour_area = largest_contour_areas[j]
                    min_contour_index = j

    return tuple(largest_contours)


def getCornerstones(maze_hsv):
    magenta_thresh_lower = (140, 200, 180)
    magenta_thresh_upper = (160, 255, 255)

    cyan_thresh_lower = (80, 200, 180)
    cyan_thresh_upper = (100, 255, 255)

    magenta_contours = getColorContours(maze_hsv, magenta_thresh_lower, magenta_thresh_upper)
    cyan_contours = getColorContours(maze_hsv, cyan_thresh_lower, cyan_thresh_upper)

    cornerstone_bounds = []

    largest_magenta_contour = getLargestContours(magenta_contours, 1)[0]

    contour_poly = cv2.approxPolyDP(largest_magenta_contour, 3, True)
    bounds = cv2.boundingRect(contour_poly)
    cornerstone_bounds.append(bounds)

    largest_cyan_contours = getLargestContours(cyan_contours, 3)

    for _, contour in enumerate(largest_cyan_contours):
        contour_poly = cv2.approxPolyDP(contour, 3, True)
        bounds = cv2.boundingRect(contour_poly)
        cornerstone_bounds.append(bounds)

    # Create tuple of ellipse centers and radii from the list of bounding rectangles
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


def isPointInRegion(point, region):
    return region[0] <= point[0] <= region[2] and region[1] <= point[1] <= region[3]


def getPointRegionIndex(point, regions):
    for i, region in enumerate(regions):
        if isPointInRegion(point, region):
            return i
    return -1


def getOverheadPerspective(maze_img, cornerstone_points):
    transformed_points = np.float32(((0, 0),                                  # Top left corner
                                     (MAZE_IMAGE_WIDTH, 0),                   # Top right corner
                                     (0, MAZE_IMAGE_HEIGHT),                  # Bottom left corner
                                     (MAZE_IMAGE_WIDTH, MAZE_IMAGE_HEIGHT)))  # Bottom right corner

    # The four regions A, B, C and D which the cornerstones can occupy
    # Format for each region: (min_x, min_y, max_x, max_y)
    cornerstone_regions = np.float32(((0, 0, 337.5, 375),         # Region A
                                      (1012.5, 0, 1350, 375),     # Region B
                                      (0, 375, 337.5, 750),       # Region C
                                      (1012.5, 375, 1350, 750)))  # Region D

    magenta_region_index = getPointRegionIndex(cornerstone_points[0], cornerstone_regions)

    # Mapping from cornerstone_regions indices to transformed_points indices
    region_map = (0, 1, 2, 3)  # Map if magenta marker is in region D

    if magenta_region_index == 0:
        # Map if magenta marker is in region A
        region_map = (3, 2, 1, 0)

    ordered_corner_points = np.zeros((4, 2), np.float32)

    # Insert magenta corner point at its ordered position
    ordered_corner_points[region_map[magenta_region_index]] = np.float32(cornerstone_points[0])

    # Insert cyan corner points at their ordered positions
    for i in range(1, 4):
        region_index = getPointRegionIndex(cornerstone_points[i], cornerstone_regions)
        ordered_corner_points[region_map[region_index]] = np.float32(cornerstone_points[i])

    homography_matrix = cv2.getPerspectiveTransform(ordered_corner_points, transformed_points)

    return cv2.warpPerspective(maze_img, homography_matrix, (MAZE_IMAGE_WIDTH, MAZE_IMAGE_HEIGHT))


def getWalls(maze_hsv):
    # Threshold-filter the image to isolate the tops of the walls
    wall_thresh_lower = (15, 0, 220)
    wall_thresh_upper = (30, 255, 255)
    maze_filtered = filterImg(maze_hsv, lower=wall_thresh_lower, upper=wall_thresh_upper)

    # Convert filtered image to grayscale, then binary
    maze_filtered_gray = cv2.cvtColor(maze_filtered, cv2.COLOR_RGB2GRAY)
    filtered_binary = getBinaryImg(maze_filtered_gray, 128, 255)

    # Retrieve the set of "blobs" in the binary image
    num_blobs, blob_img, blob_info, _ = cv2.connectedComponentsWithStats(filtered_binary)
    blob_sizes = blob_info[:, -1]

    min_blob_size = 160  # Minimum blob size in pixels for one to be considered a wall

    # Iterate through each blob, except for the first, which is the image background
    wall_mask = np.zeros(filtered_binary.shape, np.uint8)
    for i in range(1, num_blobs):
        # Filter out any blob below the minimum size to ignore the corner blocks
        if blob_sizes[i] >= min_blob_size:
            # Blobs above min size are walls, mark these in white
            wall_mask[blob_img == i] = 255

    if DEBUG:
        showImg(maze_filtered, "Filtered maze")
        showImg(cv2.cvtColor(maze_filtered_gray, cv2.COLOR_GRAY2RGB), "Filtered maze, grayscale")
        showImg(cv2.cvtColor(wall_mask, cv2.COLOR_GRAY2RGB), "Wall binary mask - output")

    return wall_mask


def markWalls(maze_rgb, wall_mask):
    blue_rgb = (0, 0, 255)

    # Stack wall image over maze image to mark walls in blue
    maze_marked = maze_rgb.copy()
    maze_marked[wall_mask == 255] = blue_rgb

    return maze_marked


def findTarget(maze_rgb, target_rgb):
    # Apply an HSV threshold to filter the image for red regions
    maze_hsv = cv2.cvtColor(maze_rgb, cv2.COLOR_RGB2HSV)
    maze_filtered = filterImg(maze_hsv, lower=(160, 100, 50), upper=(180, 255, 255))

    # Convert filtered image to grayscale, then dilate the filtered regions to ensure that
    # The entire target image is visible
    maze_filtered_gray = cv2.cvtColor(maze_filtered, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((80, 80), np.uint8)
    maze_filtered_gray_dilated = cv2.morphologyEx(maze_filtered_gray, cv2.MORPH_DILATE, kernel)

    # Convert filtered, dilated image to a binary mask
    mask = getBinaryImg(maze_filtered_gray_dilated, 50, 255)

    # Apply the mask to the original image to cover regions that don't contain the target
    maze_masked = cv2.bitwise_and(maze_rgb, maze_rgb, mask=mask)
    showImg(maze_masked, "masked maze for target matching")

    # Apply pattern matching with the sample target image to the masked image to find the target
    match_result = cv2.matchTemplate(maze_masked, target_rgb, cv2.TM_CCORR_NORMED)
    _, _, _, loc = cv2.minMaxLoc(match_result)

    # Calculate the center point and radius of the target in the image
    half_target_width = int(target_rgb.shape[0] / 2)
    center_x = int(loc[0]) + half_target_width
    center_y = int(loc[1]) + half_target_width

    return tuple((center_x, center_y, int(half_target_width / 2)))


def markTarget(maze_rgb, target_circle):
    maze_target_marked = maze_rgb.copy()
    cv2.circle(maze_target_marked, target_circle[0:2], target_circle[2], (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize('X', font, 1, 2)[0]
    text_origin = (target_circle[0] - int(text_size[0] / 2), target_circle[1] + int(text_size[1] / 2))
    cv2.putText(maze_target_marked, 'X', text_origin, font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return maze_target_marked


def detectMarker(robot_rgb):
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco_DetectorParameters.create()
    (corners, _, _) = cv2.aruco.detectMarkers(robot_rgb, aruco_dict, parameters=parameters)
    print(tuple(corners[0][0][0]))
    cv2.circle(robot_rgb, tuple(corners[0][0][0]), 20, (0, 255, 0))
    showImg(robot_rgb, "Robot with corner marked")


if __name__ == "__main__":
    maze_bgr = cv2.imread(MAZE_FILE_NAME)
    maze_rgb = cv2.cvtColor(maze_bgr, cv2.COLOR_BGR2RGB)

    showImg(maze_rgb, "Maze in RGB mode")

    maze_hsv = cv2.cvtColor(maze_rgb, cv2.COLOR_RGB2HSV)

    cornerstone_bounds = getCornerstones(maze_hsv)

    cornerstone_img = markCornerstones(maze_rgb, cornerstone_bounds)

    showImg(cornerstone_img, "Maze with cornerstones marked")

    cornerstone_points = tuple((bound[0], bound[1]) for bound in cornerstone_bounds)

    maze_overhead = getOverheadPerspective(maze_rgb, cornerstone_points)

    showImg(maze_overhead, "Maze from overhead perspective")

    wall_mask = getWalls(cv2.cvtColor(maze_overhead, cv2.COLOR_RGB2HSV))

    maze_walls_marked = markWalls(maze_overhead, wall_mask)

    showImg(maze_walls_marked, "Maze with walls marked")

    target_bgr = cv2.imread(IMAGE_LADYBUG_FILE_NAME)
    target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)

    target_circle = findTarget(maze_overhead, target_rgb)

    maze_target_marked = markTarget(maze_walls_marked, target_circle)

    showImg(maze_target_marked, "Maze with target marked")

    robot_bgr = cv2.imread(ROBOT_FILE_NAME)
    robot_rgb = cv2.cvtColor(robot_bgr, cv2.COLOR_BGR2RGB)

    detectMarker(robot_rgb)
