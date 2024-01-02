import cv2
import pyautogui
import numpy as np

# Initial screenshot
prev_screenshot = pyautogui.screenshot()
prev_screenshot = cv2.cvtColor(np.array(prev_screenshot), cv2.COLOR_RGB2BGR)

while True:
    # Capture current screenshot
    screenshot = pyautogui.screenshot()
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # Calculate the difference
    diff = cv2.absdiff(prev_screenshot, screenshot)
    
    # Apply Gaussian blur to smooth out differences
    blurred = cv2.GaussianBlur(diff, (5, 5), 0)
    
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    # Use adaptive thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours and draw rectangles
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Use a more accurate bounding rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(screenshot, [box], 0, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Highlighted Movements', screenshot)

    # Update the previous screenshot for the next iteration
    prev_screenshot = screenshot.copy()

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
