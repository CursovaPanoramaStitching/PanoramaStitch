import cv2
import numpy as np

# Load the high-resolution image
image_path = 'result\\temp_image_stitched_0000_0016.png'
image = cv2.imread(image_path)

# Initial zoom level
zoom_level = 1.0

# Create a function to handle mouse events
def select_pixels(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Check if left mouse button is clicked
        print(f"Selected pixel coordinates: (x={x}, y={y})")
        # Save the selected pixel coordinates (x, y) to a file or list

# Create a window and bind the mouse callback function
window_name = 'Image'
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, select_pixels)

# Resize the image to fit the initial zoom level
resized_image = cv2.resize(image, None, fx=zoom_level, fy=zoom_level)

# Display the image
cv2.imshow(window_name, resized_image)

# Handle keyboard events for zooming
while True:
    key = cv2.waitKey(0)
    if key == ord('q'):  # Quit if 'q' is pressed
        break
    elif key == ord('i'):  # Zoom in if 'i' is pressed
        zoom_level *= 1.1
        resized_image = cv2.resize(image, None, fx=zoom_level, fy=zoom_level)
        cv2.imshow(window_name, resized_image)
    elif key == ord('o'):  # Zoom out if 'o' is pressed
        zoom_level /= 1.1
        resized_image = cv2.resize(image, None, fx=zoom_level, fy=zoom_level)
        cv2.imshow(window_name, resized_image)

cv2.destroyAllWindows()