from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLOv8 model
model = YOLO('model/yolov8l.pt')  # Replace with the appropriate model path

# Initialize video capture
video_path = 'data/toll_gate.mp4'
cap = cv2.VideoCapture(video_path)

# Define parameters for vehicle tracking and counting
vehicle_count = 0
tracked_vehicles = {}  # Dictionary to track vehicles by ID
vehicle_id = 0
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the extended slanted gate line for counting vehicles
gate_start_point = (0, frame_height // 2 - 100)  # Start at the far left of the frame
gate_end_point = (frame_width, frame_height // 2 + 100)  # End at the far right of the frame
gate_line_color = (0, 255, 0)  # Green
gate_thickness = 2

# Define tracking parameters
max_distance = 40

# Function to check if a vehicle crosses the slanted gate
def is_crossing_gate(box_center_x, box_center_y):
    m = (gate_end_point[1] - gate_start_point[1]) / (gate_end_point[0] - gate_start_point[0])
    b = gate_start_point[1] - m * gate_start_point[0]
    gate_y_at_x = m * box_center_x + b
    return abs(box_center_y - gate_y_at_x) < 10  # Threshold to consider crossing

# Function to calculate the Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Function to get the class name based on class ID
def get_class_name(class_id):
    class_map = {2: 'CAR', 3: 'MOTORCYCLE', 5: 'BUS', 7: 'TRUCK'}
    return class_map.get(class_id, 'UNKNOWN')

# Video writer to save the output
output_path = 'vehicle_counting_output.avi'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Main loop for processing video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform vehicle detection using the YOLO model
    results = model.predict(source=frame, show=False)

    # Get detections and boxes
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()

    # Filter only vehicle detections (class 2: car, class 3: motorcycle, class 5: bus, class 7: truck)
    vehicle_indices = np.where(np.isin(class_ids, [2, 3, 5, 7]))[0]
    vehicle_boxes = boxes[vehicle_indices]
    vehicle_confidences = confidences[vehicle_indices]
    vehicle_class_ids = class_ids[vehicle_indices]

    # Create a list to store new tracking information
    new_tracked_vehicles = {}

    # Assign IDs to detected vehicles
    for i, box in enumerate(vehicle_boxes):
        x1, y1, x2, y2 = map(int, box)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        class_id = int(vehicle_class_ids[i])
        class_name = get_class_name(class_id)

        # Determine if the vehicle is already being tracked
        matched_id = None
        min_distance = float('inf')

        for vid, (vx, vy, vclass, counted) in tracked_vehicles.items():
            if vclass == class_name:
                distance = euclidean_distance((center_x, center_y), (vx, vy))
                if distance < max_distance and distance < min_distance:
                    min_distance = distance
                    matched_id = vid

        # Assign a new ID if no match found
        if matched_id is None:
            matched_id = vehicle_id
            vehicle_id += 1

        # Check if the vehicle is crossing the gate
        counted = tracked_vehicles.get(matched_id, (None, None, class_name, False))[3]
        if is_crossing_gate(center_x, center_y) and not counted:
            vehicle_count += 1
            counted = True

        # Update the list of tracked vehicles
        new_tracked_vehicles[matched_id] = (center_x, center_y, class_name, counted)

    # Update tracked vehicles with new data
    tracked_vehicles = new_tracked_vehicles

    # Draw gate line on the frame
    cv2.line(frame, gate_start_point, gate_end_point, gate_line_color, gate_thickness)
    cv2.putText(frame, f'Vehicles Counted: {vehicle_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Draw bounding boxes and IDs on the frame
    for i, box in enumerate(vehicle_boxes):
        x1, y1, x2, y2 = map(int, box)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        class_id = int(vehicle_class_ids[i])
        class_name = get_class_name(class_id)
        matched_id = next((vid for vid, data in tracked_vehicles.items() if data[2] == class_name), None)

        if matched_id is not None:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'{class_name} #{matched_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    # Write the frame to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow('Vehicle Counting', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
