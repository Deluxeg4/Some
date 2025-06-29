import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO


def run_yolo_detection(model_path, img_source, min_thresh=0.5, user_res=None, record=False):
    """
    รันการตรวจจับวัตถุด้วย YOLO บนแหล่งที่มาที่ระบุ (รูปภาพ, โฟลเดอร์, วิดีโอ หรือกล้อง USB)

    อาร์กิวเมนต์:
        model_path (str): พาธไปยังไฟล์โมเดล YOLO (เช่น "runs/detect/train/weights/best.pt")
        img_source (str): ไฟล์รูปภาพ, โฟลเดอร์, ไฟล์วิดีโอ หรือดัชนีกล้อง USB (เช่น "usb0")
        min_thresh (float): เกณฑ์ความมั่นใจขั้นต่ำ (ค่าเริ่มต้น: 0.5)
        user_res (str): ความละเอียด กว้างxสูง (เช่น "640x480") สำหรับการปรับขนาดและบันทึก (ไม่บังคับ)
        record (bool): บันทึกผลลัพธ์จากวิดีโอ/USB ไปยัง "demo1.avi" (ค่าเริ่มต้น: False)
    """

    # ตรวจสอบโมเดล
    if not os.path.exists(model_path):
        print(f'ERROR: ไม่พบโมเดลที่ {model_path}.')
        return

    # โหลดโมเดล YOLO
    model = YOLO(model_path, task='detect')
    model.to('cpu')
    labels = model.names

    # กำหนดประเภทแหล่งที่มาของอินพุต
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
            print(f'นามสกุลไฟล์ไม่รองรับ: {ext}')
            return
    elif 'usb' in img_source:
        source_type = 'usb'
        try:
            usb_idx = int(img_source[3:])
        except ValueError:
            print(f'รูปแบบแหล่งที่มา USB ไม่ถูกต้อง: {img_source}. คาดหวัง "usbX".')
            return
    else:
        print(f'แหล่งที่มาไม่ถูกต้อง: {img_source}')
        return

    # จัดการความละเอียด
    resize = False
    resW, resH = 0, 0
    if user_res:
        try:
            resW, resH = map(int, user_res.split('x'))
            resize = True
        except ValueError:
            print(f'รูปแบบความละเอียดไม่ถูกต้อง: {user_res}. คาดหวัง "WxH".')
            return

    # ตั้งค่าตัวบันทึก (recorder)
    recorder = None
    if record:
        if source_type not in ['video', 'usb']:
            print('การบันทึกรองรับเฉพาะวิดีโอ/USB เท่านั้น.')
            record = False  # ปิดการบันทึกหากไม่สามารถใช้ได้
        if not user_res:
            print('การบันทึกต้องระบุ --resolution.')
            record = False  # ปิดการบันทึกหากไม่ได้ตั้งค่าความละเอียด
        if record:
            recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH))
            if not recorder.isOpened():
                print("ข้อผิดพลาด: ไม่สามารถเปิดตัวเขียนวิดีโอได้.")
                recorder = None  # ตรวจสอบให้แน่ใจว่า recorder เป็น None หากเปิดไม่สำเร็จ

    # โหลดแหล่งที่มา
    imgs_list = []
    cap = None
    if source_type == 'image':
        imgs_list = [img_source]
    elif source_type == 'folder':
        imgs_list = sorted(
            [f for f in glob.glob(os.path.join(img_source, '*')) if os.path.splitext(f)[1].lower() in img_exts])
        if not imgs_list:
            print(f"ไม่พบรูปภาพในโฟลเดอร์: {img_source}")
            return
    elif source_type in ['video', 'usb']:
        cap = cv2.VideoCapture(img_source if source_type == 'video' else usb_idx)
        if not cap.isOpened():
            print(f"ข้อผิดพลาด: ไม่สามารถเปิดแหล่งวิดีโอหรือกล้อง USB: {img_source}")
            return
        if user_res:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

    # โทนสี
    bbox_colors = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133),
                   (88, 159, 106), (96, 202, 231), (159, 124, 168), (169, 162, 241),
                   (98, 118, 150), (172, 176, 184)]

    frame_rate_buffer = []
    fps_avg_len = 200
    img_count = 0
    avg_fps = 0.0

    # Initialize overall counts for folder/image sources outside the loop
    # เพื่อสะสมจำนวนวัตถุรวมเมื่อประมวลผลรูปภาพจากโฟลเดอร์หรือรูปภาพเดี่ยว
    overall_object_counts = {
        'bottle-can': 0,
        'bottle-glass': 0,
        'bottle-plastic': 0,
        'cap': 0
        # คุณสามารถเพิ่มคลาสอื่นๆ ที่โมเดลของคุณตรวจจับได้ที่นี่
    }

    # Main loop
    while True:
        t_start = time.perf_counter()

        # Get frame
        frame = None
        if source_type in ['image', 'folder']:
            if img_count >= len(imgs_list):
                print('ประมวลผลรูปภาพทั้งหมดแล้ว.')
                break
            frame = cv2.imread(imgs_list[img_count])
            if frame is None:
                print(f"คำเตือน: ไม่สามารถอ่านรูปภาพ {imgs_list[img_count]} ได้")
                img_count += 1
                continue
            img_count += 1
        else:
            ret, frame = cap.read()
            if not ret or frame is None:
                print('สิ้นสุดวิดีโอหรือกล้องหลุดการเชื่อมต่อ.')
                break

        # Resize if needed
        if resize:
            frame = cv2.resize(frame, (resW, resH))

        # Inference
        results = model(frame, verbose=False)
        detections = results[0].boxes

        # Initialize counts for current frame
        current_frame_object_counts = {
            'bottle-can': 0,
            'bottle-glass': 0,
            'bottle-plastic': 0,
            'cap': 0
        }
        total_objects_detected_in_frame = 0

        # Draw detections
        for det in detections:
            xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
            classid = int(det.cls.item())
            conf = det.conf.item()

            # ตรวจสอบว่าความมั่นใจสูงกว่าเกณฑ์ที่กำหนด
            if conf > min_thresh:
                class_name = labels[classid]  # ได้ชื่อคลาสจาก classid

                # เพิ่มการนับสำหรับแต่ละคลาสในเฟรมปัจจุบัน
                if class_name in current_frame_object_counts:
                    current_frame_object_counts[class_name] += 1

                # สะสมจำนวนวัตถุรวมทั้งหมด (สำหรับ source_type 'image'/'folder')
                if source_type in ['image', 'folder'] and class_name in overall_object_counts:
                    overall_object_counts[class_name] += 1

                total_objects_detected_in_frame += 1  # นับรวมวัตถุทั้งหมดที่แสดงผลในเฟรมปัจจุบัน

                # วาดสี่เหลี่ยมผืนผ้าและป้ายกำกับ
                color = bbox_colors[classid % len(bbox_colors)]
                cv2.rectangle(frame, tuple(xyxy[:2]), tuple(xyxy[2:]), color, 2)
                label = f"{class_name}: {int(conf * 100)}%"
                cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # FPS calculation
        t_stop = time.perf_counter()
        fps = 1 / (t_stop - t_start)
        frame_rate_buffer.append(fps)
        if len(frame_rate_buffer) > fps_avg_len:
            frame_rate_buffer.pop(0)
        avg_fps = np.mean(frame_rate_buffer)

        # Display information on the frame (primarily for video/usb sources)
        # แสดงข้อมูลบนเฟรม (ส่วนใหญ่สำหรับแหล่งที่มาเป็นวิดีโอ/USB)
        # สำหรับรูปภาพ/โฟลเดอร์ ค่ารวมสุดท้ายจะแสดงในคอนโซล
        if source_type in ['video', 'usb']:
            y_offset = 20  # ตำแหน่งเริ่มต้นสำหรับ FPS
            cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            y_offset += 25  # เลื่อนลงมาสำหรับบรรทัดถัดไป
            cv2.putText(frame, f"Total Objects: {total_objects_detected_in_frame}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)

            # แสดงจำนวนของแต่ละคลาสที่คุณสนใจในเฟรมปัจจุบัน
            for class_name, count in current_frame_object_counts.items():
                if count > 0:  # แสดงเฉพาะคลาสที่มีวัตถุ
                    y_offset += 25
                    cv2.putText(frame, f"{class_name}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 255), 2)

        # For image/folder source, print current frame counts to console
        # สำหรับแหล่งที่มาเป็นรูปภาพ/โฟลเดอร์ ให้พิมพ์จำนวนวัตถุของเฟรมปัจจุบันลงคอนโซล
        if source_type in ['image', 'folder']:
            print(f"\n--- Results for {imgs_list[img_count - 1]}: ---")
            print(f"Total Objects in frame: {total_objects_detected_in_frame}")
            for class_name, count in current_frame_object_counts.items():
                if count > 0:
                    print(f"  {class_name}: {count}")

        cv2.imshow("YOLO Detection", frame)

        if record and recorder:
            recorder.write(frame)

        # Delay for image/folder (3 seconds), short delay for video/usb (5 ms)
        key = cv2.waitKey(10 if source_type in ['image', 'folder'] else 5)
        if key in [ord('q'), ord('Q')]:
            break
        elif key in [ord('s'), ord('S')]:
            cv2.waitKey()  # Pause until any key is pressed again
        elif key in [ord('p'), ord('P')]:
            cv2.imwrite("capture.png", frame)
            print("บันทึกเฟรมไปที่ capture.png แล้ว")

    # Cleanup
    print(f"\n--- สรุปผลการตรวจจับทั้งหมด ---")
    print(f"FPS เฉลี่ย: {avg_fps:.2f}")

    # Print the overall accumulated counts for image/folder sources
    # พิมพ์จำนวนวัตถุรวมทั้งหมดที่สะสมมาสำหรับแหล่งที่มาเป็นรูปภาพ/โฟลเดอร์
    if source_type in ['image', 'folder']:
        print("\nจำนวนวัตถุที่ตรวจพบทั้งหมด:")
        for class_name, count in overall_object_counts.items():
            print(f"  {class_name}: {count} อัน")

    if cap:
        cap.release()
    if recorder:
        recorder.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # This block allows you to run the script from the command line as before,
    # or you can directly call the run_yolo_detection function.

    # กำหนดและแยกวิเคราะห์อาร์กิวเมนต์จากผู้ใช้สำหรับการรันผ่านบรรทัดคำสั่ง
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        help='พาธไปยังไฟล์โมเดล YOLO (เช่น "runs/detect/train/weights/best.pt")')
    parser.add_argument('--source', required=True,
                        help='ไฟล์รูปภาพ, โฟลเดอร์, ไฟล์วิดีโอ หรือดัชนีกล้อง USB (เช่น "usb0")')
    parser.add_argument('--thresh', default=0.5, type=float, help='เกณฑ์ความมั่นใจขั้นต่ำ (เช่น "0.4")')
    parser.add_argument('--resolution', default=None, help='ความละเอียด กว้างxสูง (เช่น "640x480")')
    parser.add_argument('--record', action='store_true', help='บันทึกผลลัพธ์จากวิดีโอ/USB ไปยัง "demo1.avi"')
    args = parser.parse_args()

    # ตัวอย่างการเรียกใช้ฟังก์ชันจากอาร์กิวเมนต์บรรทัดคำสั่ง
    # โลจิกของสคริปต์เดิมตอนนี้อยู่ใน run_yolo_detection แล้ว
    run_yolo_detection(args.model, args.source, args.thresh, args.resolution, args.record)

    # --- วิธีเรียกใช้ฟังก์ชันโดยตรงสำหรับกรณีการใช้งานเฉพาะของคุณ ---
    # หากต้องการรันด้วยโฟลเดอร์รูปภาพ เช่น C:/Users/bnx22/PycharmProjects/Some/image/
    # และโมเดลชื่อ 'detectModel.pt' ในไดเรกทอรีปัจจุบัน:

    # model_path_example = 'detectModel.pt'
    # image_folder_path = 'C:/Users/bnx22/PycharmProjects/Some/image/'
    #
    # if os.path.exists(model_path_example) and os.path.isdir(image_folder_path):
    #     print(f"\n--- กำลังรันการตรวจจับบนโฟลเดอร์รูปภาพ: {image_folder_path} ---")
    #     run_yolo_detection(
    #         model_path=model_path_example,
    #         img_source=image_folder_path,
    #         min_thresh=0.6,  # ตัวอย่างเกณฑ์
    #         user_res="800x600", # ตัวอย่างความละเอียด
    #         record=False
    #     )
    # else:
    #     print(f"\n--- ข้ามตัวอย่างการเรียกใช้ฟังก์ชันโดยตรง: ไม่พบโมเดล ({model_path_example}) หรือโฟลเดอร์ ({image_folder_path}) ---")

    # คุณสามารถยกเลิกคอมเมนต์และแก้ไขส่วนด้านบนเพื่อทดสอบการเรียกใช้ฟังก์ชันโดยตรงได้