import cv2
import numpy as np

def canny(img):
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(canny):
    height = canny.shape[0]
    width = canny.shape[1]
    mask = np.zeros_like(canny)
    triangle = np.array([[
        (200, height),
        (800, 350),
        (1200, height)
    ]], np.int32)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image

def houghLines(cropped_canny):
    return cv2.HoughLinesP(cropped_canny, 2, np.pi / 180, 100,
                           np.array([]), minLineLength=40, maxLineGap=5)

def addWeighted(frame, line_image):
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)

def display_lines(img, lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (100, 100, 255), 10)
    return line_image

def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])
    y2 = int(y1 * 3.0 / 5)
    if slope == 0:  # Avoid division by zero
        slope = 0.1
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    
    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_points(image, left_fit_average)
    else:
        left_line = None
    
    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_points(image, right_fit_average)
    else:
        right_line = None
    
    if left_line is not None and right_line is not None:
        averaged_lines = [left_line, right_line]
    else:
        averaged_lines = None
    
    return averaged_lines

def draw_lane_polygon(image, lines):
    if lines is None:
        return image

    # Get the points for the left and right lines
    left_line, right_line = lines

    # Extract the x and y coordinates for both lines
    x1_left, y1_left, x2_left, y2_left = left_line[0]
    x1_right, y1_right, x2_right, y2_right = right_line[0]

    # Define points to create a polygon between the lines
    pts = np.array([
        [x1_left, y1_left], 
        [x2_left, y2_left],
        [x2_right, y2_right], 
        [x1_right, y1_right]
    ], np.int32)

    # Reshape points for fillPoly
    pts = pts.reshape((-1, 1, 2))

    # Draw a light blue filled polygon on the lane
    cv2.fillPoly(image, [pts], (105, 225, 225))

    return image

cap = cv2.VideoCapture("D:/vehicle_helper_[pro/test1.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    canny_image = canny(frame)
    if canny_image is None:
        break

    cropped_canny = region_of_interest(canny_image)
    lines = houghLines(cropped_canny)
    averaged_lines = average_slope_intercept(frame, lines)
    
    if averaged_lines is not None:
        line_image = display_lines(frame, averaged_lines)
        combo_image = addWeighted(frame, line_image)

        # Highlight the detected lane with a light blue polygon
        highlighted_lane = draw_lane_polygon(combo_image, averaged_lines)

        cv2.imshow("result", highlighted_lane)
    else:
        cv2.imshow("result", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
