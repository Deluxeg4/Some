import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to YOLO model file (e.g., "runs/detect/train/weights/best.pt")')
parser.add_argument('--source', required=True, help='Image file, folder, video file, or USB camera index (e.g., "usb0")')
parser.add_argument('--thresh', default=0.5, help='Minimum confidence threshold (e.g., "0.4")')
parser.add_argument('--resolution', default=None, help='Resolution WxH (e.g., "640x480")')
parser.add_argument('--record', action='store_true', help='Record results from video/USB to "demo1.avi"')
args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

# Check model
if not os.path.exists(model_path):
    print('ERROR: Model not found.')
    sys.exit(0)

# Load YOLO model
model = YOLO(model_path, task='detect')
model.to('cpu')
labels = model.names

# Determine input source type
img_exts = ['.jpg', '.jpeg', '.png', '.bmp']
vid_exts = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    ext = os.path.splitext(img_source)[1].lower()
    if ext in img_exts:
        source_type = 'image'
    elif ext in vid_exts:
        source_type = 'video'
    else:
        print(f'Unsupported file extension: {ext}')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
else:
    print(f'Invalid source: {img_source}')
    sys.exit(0)

# Handle resolution
resize = False
if user_res:
    resW, resH = map(int, user_res.split('x'))
    resize = True

# Set up recorder
if record:
    if source_type not in ['video', 'usb']:
        print('Recording only supported for video/usb.')
        sys.exit(0)
    if not user_res:
        print('Recording requires --resolution.')
        sys.exit(0)
    recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH))

# Load source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(img_source + '/*') if os.path.splitext(f)[1].lower() in img_exts]
elif source_type in ['video', 'usb']:
    cap = cv2.VideoCapture(img_source if source_type == 'video' else usb_idx)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)

# Color scheme
bbox_colors = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133),
               (88, 159, 106), (96, 202, 231), (159, 124, 168), (169, 162, 241),
               (98, 118, 150), (172, 176, 184)]

frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Main loop
while True:
    t_start = time.perf_counter()

    # Get frame
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('All images processed.')
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
    else:
        ret, frame = cap.read()
        if not ret or frame is None:
            print('End of video or camera disconnected.')
            break

    # Resize if needed
    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Inference
    results = model(frame, verbose=False)
    detections = results[0].boxes
    object_count = 0

    # Draw detections
    for det in detections:
        xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
        classid = int(det.cls.item())
        conf = det.conf.item()
        if conf > min_thresh:
            color = bbox_colors[classid % len(bbox_colors)]
            cv2.rectangle(frame, tuple(xyxy[:2]), tuple(xyxy[2:]), color, 2)
            label = f"{labels[classid]}: {int(conf*100)}%"
            cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            object_count += 1

    # FPS calculation
    t_stop = time.perf_counter()
    fps = 1 / (t_stop - t_start)
    frame_rate_buffer.append(fps)
    if len(frame_rate_buffer) > fps_avg_len:
        frame_rate_buffer.pop(0)
    avg_fps = np.mean(frame_rate_buffer)

    if source_type in ['video', 'usb']:
        cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Objects: {object_count}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("YOLO Detection", frame)
    if record:
        recorder.write(frame)

    key = cv2.waitKey(0 if source_type in ['image', 'folder'] else 5)
    if key in [ord('q'), ord('Q')]:
        break
    elif key in [ord('s'), ord('S')]:
        cv2.waitKey()
    elif key in [ord('p'), ord('P')]:
        cv2.imwrite("capture.png", frame)

# Cleanup
print(f"Average FPS: {avg_fps:.2f}")
if source_type in ['video', 'usb']:
    cap.release()
if record:
    recorder.release()
cv2.destroyAllWindows()

