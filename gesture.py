import cv2
import numpy as np

# Load the desired gesture representation
gesture_representation = cv2.imread("test_image1.jpg", cv2.IMREAD_GRAYSCALE)  

# Load the test video
cap = cv2.VideoCapture("test_video1.mov")  

# Define font and text color
font = cv2.FONT_HERSHEY_SIMPLEX
color = (0, 255, 0)  # Bright green color

# Define the method for comparison
method = cv2.TM_CCOEFF_NORMED

# Set a threshold for detection
threshold = 0.5



# Loop through the video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform template matching
    result = cv2.matchTemplate(gray_frame, gesture_representation, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # If the maximum correlation value is above the threshold, gesture is detected
    if max_val >= threshold:
        # Get the coordinates of the detected region
        top_left = max_loc
        bottom_right = (top_left[0] + gesture_representation.shape[1], top_left[1] + gesture_representation.shape[0])

        # Draw a rectangle around the detected region
        cv2.rectangle(frame, top_left, bottom_right, color, 2)

        # Annotate the frame with "DETECTED" text
        text = "DETECTED"
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = frame.shape[1] - text_size[0] - 10
        text_y = text_size[1] + 10
        cv2.putText(frame, text, (text_x, text_y), font, 1, color, 2)

    # Display the frame
    cv2.imshow('Frame', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
