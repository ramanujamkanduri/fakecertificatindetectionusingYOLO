from ultralytics import YOLO
import cv2

# Define the file paths
file_path = r"C:\Users\hp\OneDrive\Documents\YOLOFinetunedonUniv.pt"
input_image_path = r"C:\Users\hp\OneDrive\Documents\tenth.jpg"
output_image_path = "C:/Users/hp/OneDrive/Documents/second_detection.jpg"

print("File path:", file_path)

# Initialize the YOLO model for both main object detection and certificate detection
model = YOLO(file_path)
certificate_detection_model = YOLO(file_path)

# Perform prediction and get the results for main object detection
results = model.predict(input_image_path, conf=0.3, iou=0.5, show=True)

# Detect if the image contains a fake certificate
image = cv2.imread(input_image_path)
certificate_detection_results = certificate_detection_model.predict(image)

is_fake_certificate = False
for detection in certificate_detection_results[0].boxes.data:
    class_id = int(detection[5])
    if class_id == 0:  # Assuming class 0 represents a fake certificate
        is_fake_certificate = True
        break

# Save the object-detected image
results[0].save(output_image_path)

# Print the result
if is_fake_certificate:
    print("The image contains a fake certificate.")
else:
    print("The image does not contain a fake certificate.")

print(f"Object-detected image saved to: {output_image_path}")