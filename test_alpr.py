from fast_alpr.alpr import ALPR
import cv2

# Initialize the ALPR pipeline
alpr = ALPR(
    detector_model="yolo-v9-t-640-license-plate-end2end", detector_conf_thresh=0.1
)

# Open a video file or use a webcam (0 for default webcam)
cap = cv2.VideoCapture("Centrale_demo.mp4")  # Replace with your video path or 0 for webcam

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))

# Create a resizable window
cv2.namedWindow("ALPR Result", cv2.WINDOW_NORMAL)

def blur_license_plates(frame, alpr_results):
    """
    Blurs the regions of the frame where license plates are detected.

    Parameters:
        frame (numpy.ndarray): The input frame.
        alpr_results (list[ALPRResult]): List of ALPR results containing detection and OCR info.

    Returns:
        numpy.ndarray: The frame with blurred license plates.
    """
    for result in alpr_results:
        detection = result.detection
        bbox = detection.bounding_box
        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2

        # Extract the region of interest (ROI) containing the license plate
        roi = frame[y1:y2, x1:x2]

        # Apply Gaussian blur to the ROI
        blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)

        # Replace the ROI in the original frame with the blurred ROI
        frame[y1:y2, x1:x2] = blurred_roi

    return frame

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if no more frames

    # Perform ALPR on the frame
    alpr_results = alpr.predict(frame)

    # Blur the license plates in the frame
    blurred_frame = blur_license_plates(frame.copy(), alpr_results)

    # Draw predictions on the blurred frame (optional)
    annotated_frame = alpr.draw_predictions(blurred_frame)

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    # Display the annotated frame in the resizable window
    cv2.imshow("ALPR Result", annotated_frame)

    # Exit on key press (e.g., 'q' key)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()