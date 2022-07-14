import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plot

MAZE_FILE_NAME = "../Maze.png"
ROBOT_FILE_NAME = "../Robot.png"
IMAGE_LADYBUG_FILE_NAME = "../Ladybug_small.png"
MAP_FILE_NAME = "../MapBuilt.png"

if __name__ == "__main__":
    maze_bgr = cv2.imread(MAZE_FILE_NAME)
    maze_rgb = cv2.cvtColor(maze_bgr, cv2.COLOR_BGR2RGB)

    maze_hsv = cv2.cvtColor(maze_bgr, cv2.COLOR_BGR2HSV)

    cyan_thresh_lower = np.array([89, 230, 230])
    cyan_thresh_upper = np.array([90, 255, 255])
    maze_cyan_mask = cv2.inRange(maze_hsv, cyan_thresh_lower, cyan_thresh_upper)
    maze_cyan = cv2.bitwise_and(maze_rgb, maze_rgb, mask=maze_cyan_mask)
    maze_cyan_gray = cv2.cvtColor(maze_cyan, cv2.COLOR_RGB2GRAY)

    fig2, axes2 = plt.subplots(figsize=(9, 5))
    axes2.imshow(maze_cyan)
    axes2.set_title("Maze with cyan filter applied")
    fig2.show()

    magenta_thresh_lower = np.array([140, 230, 230])
    magenta_thresh_upper = np.array([160, 255, 255])
    maze_magenta_mask = cv2.inRange(maze_hsv, magenta_thresh_lower, magenta_thresh_upper)
    maze_magenta = cv2.bitwise_and(maze_rgb, maze_rgb, mask=maze_magenta_mask)
    maze_magenta_gray = cv2.cvtColor(maze_magenta, cv2.COLOR_RGB2GRAY)

    fig3, axes3 = plt.subplots(figsize=(9, 5))
    axes3.imshow(maze_magenta)
    axes3.set_title("Maze with magenta filter applied")
    fig3.show()

    kernel = np.ones((8, 8), np.uint8)

    circle_accumulation = 0.8
    circle_min_dist = 50
    circle_param_1 = 5
    circle_param_2 = 10
    circle_min_radius = 3
    circle_max_radius = 20

    maze_cyan_gray_closed = cv2.morphologyEx(maze_cyan_gray, cv2.MORPH_CLOSE, kernel)
    fig4, axes4 = plt.subplots(figsize=(9, 5))
    axes4.imshow(cv2.cvtColor(maze_cyan_gray_closed, cv2.COLOR_GRAY2RGB))
    axes4.set_title("Grayscale, cyan-filtered, closed maze")
    fig4.show()
    cyan_circles = cv2.HoughCircles(maze_cyan_gray_closed,
                                    cv2.HOUGH_GRADIENT,
                                    circle_accumulation,
                                    circle_min_dist,
                                    param1=circle_param_1,
                                    param2=circle_param_2,
                                    minRadius=circle_min_radius,
                                    maxRadius=circle_max_radius)

    if cyan_circles is None:
        print(f"Couldn't find any cyan circles")
    else:
        print(f"Found {len(cyan_circles[0])} cyan circles:")
        for circle in cyan_circles[0]:
            cv2.circle(maze_rgb, (circle[0], circle[1]), circle[2], (255, 0, 255), 2)
            # print(f"\tFound cyan circle of radius {circle[2]} at: ({circle[0]}, {circle[1]})")

    maze_magenta_gray_closed = cv2.morphologyEx(maze_magenta_gray, cv2.MORPH_CLOSE, kernel)
    fig5, axes5 = plt.subplots(figsize=(9, 5))
    axes5.imshow(cv2.cvtColor(maze_magenta_gray_closed, cv2.COLOR_GRAY2RGB))
    axes5.set_title("Grayscale, magenta-filtered, closed maze")
    fig5.show()
    magenta_circles = cv2.HoughCircles(maze_magenta_gray_closed,
                                       cv2.HOUGH_GRADIENT,
                                       circle_accumulation,
                                       circle_min_dist,
                                       param1=circle_param_1,
                                       param2=circle_param_2,
                                       minRadius=circle_min_radius,
                                       maxRadius=circle_max_radius)

    if magenta_circles is None:
        print(f"Couldn't find any magenta circles")
    else:
        print(f"Found {len(magenta_circles[0])} magenta circles:")
        for circle in magenta_circles[0]:
            cv2.circle(maze_rgb, (circle[0], circle[1]), circle[2], (0, 255, 255), 2)
            # print(f"\tFound magenta circle of radius {circle[2]} at: ({circle[0]}, {circle[1]})")

    fig, axes = plt.subplots(figsize=(9, 5))
    axes.imshow(maze_rgb)
    axes.set_title("Maze in RGB mode")
    fig.show()
