import cv2
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

DEBUG = False
MAZE_FILE_NAME = "../Maze_2.png"
ROBOT_FILE_NAME = "../Robot_2.png"
IMAGE_LADYBUG_FILE_NAME = "../Ladybug_small.png"
MAP_FILE_NAME = "../MapBuilt.png"
MAZE_IMAGE_WIDTH_RATIO = 9
MAZE_IMAGE_HEIGHT_RATIO = 5
MAZE_IMAGE_WIDTH = 900
MAZE_IMAGE_HEIGHT = 500
MAZE_CELL_SIZE = 100


class Direction(Enum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


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


def getHomographyMatrix(cornerstone_points):
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

    return cv2.getPerspectiveTransform(ordered_corner_points, transformed_points)


def getHomographyZRotation(matrix):
    # Rotation along the z-axis caused by a homography transform
    return -np.arctan2(matrix[0, 1], matrix[0, 0])


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


def getWallLocations(wall_mask):
    non_zero_pixels = cv2.findNonZero(wall_mask)

    num_rows = 2 * MAZE_IMAGE_HEIGHT_RATIO - 1
    num_cols = MAZE_IMAGE_WIDTH_RATIO

    # 2D array of bools, where True indicates the presence of a wall at the row and column
    # indicated by the first and second indices, respectively
    wall_locations = np.zeros((num_rows, num_cols), dtype=bool)

    # Number of pixels either side of a cell boundary to check for a potential wall
    wall_width_margin = 10

    # Number of pixels in from each cell corner to begin checking for a potential wall
    wall_length_margin = 20

    for i in range(num_rows):
        for j in range(num_cols):
            # Bounds for East-West walls
            wall_x_min = MAZE_CELL_SIZE * j + wall_length_margin
            wall_x_max = MAZE_CELL_SIZE * (j + 1) - wall_length_margin
            wall_y_min = MAZE_CELL_SIZE * int((i + 1) / 2) - wall_width_margin
            wall_y_max = wall_y_min + 2 * wall_width_margin

            # If the walls run North-South in this row, check for North-South walls
            if i % 2 == 0:
                wall_x_min = MAZE_CELL_SIZE * (j + 1) - wall_width_margin
                wall_x_max = wall_x_min + 2 * wall_width_margin
                wall_y_min = MAZE_CELL_SIZE * int(i / 2) + wall_length_margin
                wall_y_max = MAZE_CELL_SIZE * int(i / 2 + 1) - wall_length_margin

                # For a North-South wall row, the final value is always True (always a wall at the right edge)
                wall_locations[i][-1] = True

            for pixel in non_zero_pixels:
                # A wall exists at this row and column if there is a non-zero pixel in the region
                if wall_x_min < pixel[0][0] < wall_x_max and wall_y_min < pixel[0][1] < wall_y_max:
                    wall_locations[i][j] = True
                    break  # Wall found, don't continue to loop through all pixels

    return wall_locations


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

    if DEBUG:
        showImg(maze_masked, "masked maze for target matching")

    target_center = getMatchCenter(target_rgb, maze_masked)

    return tuple((target_center[0], target_center[1], 25))


def markTarget(maze_rgb, target_circle):
    maze_target_marked = maze_rgb.copy()
    cv2.circle(maze_target_marked, target_circle[0:2], target_circle[2], (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize('X', font, 1, 2)[0]
    text_origin = (target_circle[0] - int(text_size[0] / 2), target_circle[1] + int(text_size[1] / 2))
    cv2.putText(maze_target_marked, 'X', text_origin, font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return maze_target_marked


def getRobotDirection(robot_rgb, homography_rotation):
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco_DetectorParameters.create()

    (corners, _, _) = cv2.aruco.detectMarkers(robot_rgb, aruco_dict, parameters=parameters)

    # First corner in the corner list is the corner pointing to the robot's front side
    front_corner = tuple(corners[0][0][0])

    # Find robot's center by getting the average x and y coords of the marker corners
    robot_center_x = sum(corner[0] for corner in corners[0][0]) * 0.25
    robot_center_y = sum(corner[1] for corner in corners[0][0]) * 0.25
    robot_center = tuple((robot_center_x, robot_center_y))

    # The rotation of the front of the robot from due East, in radians
    front_angle = np.arctan2(robot_center[1] - front_corner[1], front_corner[0] - robot_center[0])

    if DEBUG:
        robot_marker = robot_rgb.copy()
        cv2.circle(robot_marker, (int(front_corner[0]), int(front_corner[1])), 5, (0, 255, 0), -1)
        cv2.circle(robot_marker, (int(robot_center[0]), int(robot_center[1])), 5, (255, 0, 0), -1)
        showImg(robot_marker, "Robot with marker center and front marked")
        print(corners)
        print(f"Front corner: {front_corner}")
        print(f"Robot center: {robot_center}")
        print(f"Homography rotation: {homography_rotation}")
        print(f"Front angle: {front_angle}")

    two_pi = 2 * np.pi
    # Get angle of robot front in transformed maze image, bounded between zero and two pi radians
    transformed_front_angle = (front_angle + homography_rotation) % two_pi

    pi_on_four = np.pi / 4
    direction = Direction.EAST  # Assume robot is facing East by default

    # Determine actual robot direction based on transformed front angle
    # Split rotation space 0 -> 2 * pi into four quadrants
    if pi_on_four <= transformed_front_angle < 3 * pi_on_four:
        direction = Direction.NORTH
    elif transformed_front_angle < 5 * pi_on_four:
        direction = Direction.WEST
    elif transformed_front_angle < 7 * pi_on_four:
        direction = Direction.SOUTH

    return direction


def discardOutlierPoints(points):
    # Modified from the top answer on this StackOverflow page:
    # https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
    x = [point[0] for point in points]
    y = [point[1] for point in points]

    d_x = np.abs(x - np.median(x))
    d_y = np.abs(y - np.median(y))

    mdev_x = np.median(d_x)
    mdev_y = np.median(d_y)

    s_x = d_x / mdev_x if mdev_x else 0.
    s_y = d_y / mdev_y if mdev_y else 0.

    filtered_points = []

    m_value = 1

    for i in range(len(points)):
        if s_x[i] < m_value and s_y[i] < m_value:
            filtered_points.append(points[i])

    return filtered_points


def getMatchCenter(query_img, train_img):
    # Get the average of the set of feature matches between query_img and train_img
    # Discard any outliers before averaging to improve accuracy
    orb = cv2.ORB_create()

    keypoints1, descriptors1 = orb.detectAndCompute(query_img, None)
    keypoints2, descriptors2 = orb.detectAndCompute(train_img, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Sort the matched features by their distance property to determine the best matches
    matches = sorted(bf.match(descriptors1, descriptors2), key=lambda x: x.distance)

    # Extract the image coordinates of the 5 best matches
    best_match_points = []

    num_best_matches = min(12, len(matches))

    if DEBUG:
        match_img = cv2.drawMatches(query_img,
                                    keypoints1,
                                    train_img,
                                    keypoints2,
                                    matches[:num_best_matches],
                                    None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        showImg(match_img, "Maze and robot, feature-matched")

    for match in matches[0:num_best_matches]:
        maze_img_index = match.trainIdx

        best_match_points.append(keypoints2[maze_img_index].pt)

    best_match_points = discardOutlierPoints(best_match_points)
    num_best_matches = len(best_match_points)

    # Take center of the robot as the average of the best match points
    center_x = int(sum(point[0] for point in best_match_points) * 1 / num_best_matches)
    center_y = int(sum(point[1] for point in best_match_points) * 1 / num_best_matches)

    return tuple((center_x, center_y))


def findRobot(maze_rgb, robot_rgb):
    # Apply an HSV threshold to filter the image for blue-green regions (unique to the robot)
    maze_hsv = cv2.cvtColor(maze_rgb, cv2.COLOR_RGB2HSV)
    maze_filtered = filterImg(maze_hsv, lower=(80, 110, 150), upper=(90, 130, 170))

    # Convert filtered image to grayscale, then dilate the filtered regions to ensure that
    # The entire target image is visible
    maze_filtered_gray = cv2.cvtColor(maze_filtered, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((50, 50), np.uint8)
    maze_filtered_gray_dilated = cv2.morphologyEx(maze_filtered_gray, cv2.MORPH_DILATE, kernel)

    # Convert filtered, dilated image to a binary mask
    mask = getBinaryImg(maze_filtered_gray_dilated, 50, 255)

    # Apply the mask to the original image to cover regions that don't contain the target
    maze_masked = cv2.bitwise_and(maze_rgb, maze_rgb, mask=mask)

    robot_rgb_scaled = cv2.resize(robot_rgb, (100, 100))
    if DEBUG:
        showImg(maze_masked, "masked maze for target matching")
        showImg(robot_rgb_scaled, "Scaled robot image")

    robot_center = getMatchCenter(robot_rgb_scaled, maze_masked)

    return tuple((robot_center[0], robot_center[1], 25))


def getCell(img_point):
    row = int(np.floor(img_point[1] / MAZE_CELL_SIZE))
    col = int(np.floor(img_point[0] / MAZE_CELL_SIZE))

    return tuple((row, col))


def getDirectionChar(direction):
    direction_char = '>'
    if direction == Direction.NORTH:
        direction_char = '^'
    elif direction == Direction.WEST:
        direction_char = '<'
    elif direction == Direction.SOUTH:
        direction_char = 'v'

    return direction_char


def markRobot(maze_rgb, robot_circle, robot_direction):
    maze_robot_marked = maze_rgb.copy()
    cv2.circle(maze_robot_marked, robot_circle[0:2], robot_circle[2], (255, 0, 0), 2)

    direction_char = getDirectionChar(robot_direction)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(direction_char, font, 1, 2)[0]
    text_origin = (robot_circle[0] - int(text_size[0] / 2), robot_circle[1] + int(text_size[1] / 2))
    cv2.putText(maze_robot_marked, direction_char, text_origin, font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    return maze_robot_marked


def getMazeString(wall_locations, robot_location, robot_direction, target_location):
    maze_string = " ---" * MAZE_IMAGE_WIDTH_RATIO + " \n"

    for i in range(MAZE_IMAGE_HEIGHT_RATIO):
        # Build a string for an empty row, then convert it to a list for easy modification
        row_list = list("|" + "    " * (MAZE_IMAGE_WIDTH_RATIO - 1) + "   |\n")

        # If robot is in this maze row, add a marker for its heading to the output string
        if robot_location[0] == i:
            row_list[4 * robot_location[1] + 2] = getDirectionChar(robot_direction)

        if target_location[0] == i:
            row_list[4 * target_location[1] + 2] = 'x'

        # Add North-South walls for this row
        for j, isWall in enumerate(wall_locations[2 * i]):
            if isWall:
                row_list[4 * (j + 1)] = '|'

        # Append the constructed row string to the maze string
        maze_string += "".join(row_list)

        # Unless row is the 5th row, add a wall row beneath it representing the East-West walls
        if i < MAZE_IMAGE_HEIGHT_RATIO - 1:
            # Build a string for an empty wall row, then convert it to a list for easy modification
            wall_list = list(' ' * (4 * MAZE_IMAGE_WIDTH_RATIO + 1) + '\n')

            # Add East-West walls for this wall row
            for j, isWall in enumerate(wall_locations[2 * i + 1]):
                if isWall:
                    wall_list[4 * j + 1] = '-'
                    wall_list[4 * j + 2] = '-'
                    wall_list[4 * j + 3] = '-'

            maze_string += "".join(wall_list)

    maze_string += " ---" * MAZE_IMAGE_WIDTH_RATIO + " \n"

    return maze_string


def getMap(maze_img, target_img, robot_img):
    maze_bgr = cv2.imread(maze_img)
    maze_rgb = cv2.cvtColor(maze_bgr, cv2.COLOR_BGR2RGB)

    maze_hsv = cv2.cvtColor(maze_rgb, cv2.COLOR_RGB2HSV)

    cornerstone_bounds = getCornerstones(maze_hsv)

    cornerstone_points = tuple((bound[0], bound[1]) for bound in cornerstone_bounds)

    homography_matrix = getHomographyMatrix(cornerstone_points)

    maze_overhead = cv2.warpPerspective(maze_rgb, homography_matrix, (MAZE_IMAGE_WIDTH, MAZE_IMAGE_HEIGHT))

    wall_mask = getWalls(cv2.cvtColor(maze_overhead, cv2.COLOR_RGB2HSV))
    wall_locations = getWallLocations(wall_mask)

    maze_walls_marked = markWalls(maze_overhead, wall_mask)

    target_bgr = cv2.imread(IMAGE_LADYBUG_FILE_NAME)
    target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)

    target_circle = findTarget(maze_overhead, target_rgb)
    target_location = getCell(target_circle[0:2])

    maze_target_marked = markTarget(maze_walls_marked, target_circle)

    robot_bgr = cv2.imread(ROBOT_FILE_NAME)
    robot_rgb = cv2.cvtColor(robot_bgr, cv2.COLOR_BGR2RGB)

    homography_rotation = getHomographyZRotation(homography_matrix)
    robot_direction = getRobotDirection(robot_rgb, homography_rotation)

    robot_circle = findRobot(maze_overhead, robot_rgb)
    robot_location = getCell(robot_circle[0:2])

    maze_robot_marked = markRobot(maze_target_marked, robot_circle, robot_direction)

    return getMazeString(wall_locations, robot_location, robot_direction, target_location)


if __name__ == "__main__":
    maze_bgr = cv2.imread(MAZE_FILE_NAME)
    maze_rgb = cv2.cvtColor(maze_bgr, cv2.COLOR_BGR2RGB)

    showImg(maze_rgb, "Maze in RGB mode")

    maze_hsv = cv2.cvtColor(maze_rgb, cv2.COLOR_RGB2HSV)

    cornerstone_bounds = getCornerstones(maze_hsv)

    cornerstone_img = markCornerstones(maze_rgb, cornerstone_bounds)

    showImg(cornerstone_img, "Maze with cornerstones marked")

    cornerstone_points = tuple((bound[0], bound[1]) for bound in cornerstone_bounds)

    homography_matrix = getHomographyMatrix(cornerstone_points)

    # Perform a perspective transform to produce an overhead view of the maze
    maze_transformed = cv2.warpPerspective(maze_rgb, homography_matrix, (MAZE_IMAGE_WIDTH, MAZE_IMAGE_HEIGHT))

    showImg(maze_transformed, "Maze from overhead perspective")

    wall_mask = getWalls(cv2.cvtColor(maze_transformed, cv2.COLOR_RGB2HSV))
    wall_locations = getWallLocations(wall_mask)

    maze_walls_marked = markWalls(maze_transformed, wall_mask)

    showImg(maze_walls_marked, "Maze with walls marked")

    target_bgr = cv2.imread(IMAGE_LADYBUG_FILE_NAME)
    target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)

    target_circle = findTarget(maze_transformed, target_rgb)
    target_location = getCell(target_circle[0:2])

    maze_target_marked = markTarget(maze_walls_marked, target_circle)

    showImg(maze_target_marked, "Maze with target marked")

    robot_bgr = cv2.imread(ROBOT_FILE_NAME)
    robot_rgb = cv2.cvtColor(robot_bgr, cv2.COLOR_BGR2RGB)

    homography_rotation = getHomographyZRotation(homography_matrix)
    robot_direction = getRobotDirection(robot_rgb, homography_rotation)

    robot_circle = findRobot(maze_transformed, robot_rgb)
    robot_location = getCell(robot_circle[0:2])

    maze_robot_marked = markRobot(maze_target_marked, robot_circle, robot_direction)
    showImg(maze_robot_marked, "Maze with robot marked")

    print(getMazeString(wall_locations, robot_location, robot_direction, target_location))
