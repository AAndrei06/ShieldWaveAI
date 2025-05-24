import cv2
from ultralytics import YOLO
import time

# Load a YOLOv8n PyTorch model
model = YOLO("yolov8n.pt")

# Export the model to OpenVINO format
model.export(format="openvino")  # creates 'yolov8n_openvino_model/'

# Load the exported OpenVINO model
ov_model = YOLO("yolov8n_openvino_model/", task='detect')

cap = cv2.VideoCapture(0)


# Variables to calculate FPS
frame_count = 0
start_time = time.time()

# Loop through the video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    results = ov_model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        current_fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()

    # Display the FPS on the frame
    cv2.putText(annotated_frame, f"FPS: {current_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the annotated frame
    cv2.imshow("YOLOv8 OpenVINO Inference", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Control FPS to 9
    time.sleep(0.1)

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

