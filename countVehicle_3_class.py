import cv2
import numpy as np
from boxmot import BYTETracker
import time
from ultralytics import YOLO
import argparse


def main(args):
    tracker = BYTETracker()
    model = YOLO(f'model/{args.name_model}.pt', task='detect')
    vid = cv2.VideoCapture(f'video/{args.name_video}.{args.type_video}')

    total_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    widthFrame = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    heightFrame = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    size = (widthFrame, heightFrame)
    out = cv2.VideoWriter('output.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    ROI = [
        np.array([[0, 976], [567, 136], [1630, 140], [1919, 426],
                  [1920, 1080], [0, 1080]], np.int32),
        np.array([]),
        np.array([]),
        np.array([[0, 442], [1151, 380], [1665, 450],
                  [1641, 1080], [0, 1080]], np.int32),
        np.array([]),
        np.array([[290, 897], [766, 897], [1080, 1452],
                  [1080, 1920], [0, 1920], [0, 1134]], np.int32),
        np.array([]),
        np.array([[716, 1080], [630, 283], [629, 219], [
            923, 219], [1920, 712], [1920, 1080]]),
        np.array([[0, 638], [728, 292], [1147, 310],
                  [915, 1080], [0, 1080]], np.int32),
        np.array([[0, 1080], [291, 861], [722, 861],
                  [1080, 1458], [1080, 1920], [0, 1920]], np.int32),
        np.array([[0, 604], [967, 600], [1080, 696],
                  [1080, 1240], [0, 1240]], np.int32),
        np.array([[0, 945], [278, 660], [830, 660], [
            1080, 932], [1080, 1616], [0, 1616]], np.int32)
    ]

    points = ROI[args.num_video - 1]
    numOfVehicle = [{} for _ in range(4)]
    total_people_look_camera = set()
    original_points = points.copy()
    sum_time = 0
    sum_time_tracker = 0
    y_count_up = 945
    y_count_down = 1080
    y_face = 1529
    while True:
        start_time = time.time()
        ret, im = vid.read()
        if not ret:
            break
        people_look_this_frame = []
        face_this_frame = []
        xmin, ymin = np.min(original_points, axis=0)
        xmax, ymax = np.max(original_points, axis=0)
        width_roi = xmax - xmin
        height_roi = ymax - ymin
        mask = np.zeros((height_roi, width_roi), dtype=np.uint8)
        im_crop = im[ymin:ymax, xmin:xmax]
        cv2.polylines(im, [original_points], True, (0, 255, 0), 2)
        points = original_points - np.array([xmin, ymin])
        cv2.fillPoly(mask, [points], 255)
        roi_img = cv2.bitwise_and(im_crop, im_crop, mask=mask)
        frame_number = vid.get(cv2.CAP_PROP_POS_FRAMES)
        time_start_predict = time.time()
        result = model.predict(roi_img, conf=args.conf_thres, imgsz=640)

        boxes = result[0].boxes
        dets = np.empty((0, 6), dtype=np.float32)
        for i in range(len(boxes.xyxy)):
            bbox = boxes.xyxy[i].cpu().numpy()
            y_down = bbox[3] + ymin
            y_up = bbox[1] + ymin
            im = cv2.line(im, (0, y_count_up),
                          (1919, y_count_up), (127, 127, 255), 1)
            im = cv2.line(im, (0, y_count_down),
                          (1919, y_count_down), (127, 127, 255), 1)
            im = cv2.line(im, (0, y_face), (1919, y_face), (127, 127, 255), 1)
            if boxes.cls[i].item() == 0 or boxes.cls[i].item() == 2:
                if y_down >= y_count_up and y_up <= y_face:
                    det = np.array([[bbox[0], bbox[1], bbox[2], bbox[3],
                                    boxes.conf[i].item(), boxes.cls[i].item()]])
                    dets = np.vstack([dets, det])
            elif boxes.conf[i].item() > 0.25:
                face_this_frame.append(
                    (int(bbox[0] + xmin), int(bbox[1] + ymin), int(bbox[2] + xmin), int(bbox[3] + ymin)))

        time_tracker_start = time.time()
        tracks = tracker.update(dets, roi_img)
        time_tracker = time.time() - time_tracker_start
        sum_time_tracker += time_tracker

        if tracks.shape[0] != 0:
            for track in tracks:
                x1, y1, x2, y2, track_id, confidence, class_id, _ = track
                x1, y1, x2, y2 = int(x1 + xmin), int(y1 + ymin), int(
                    x2 + xmin), int(y2 + ymin)
                if y2 >= y_count_up and y1 <= y_count_down:
                    numOfVehicle[int(class_id)][int(track_id)] = numOfVehicle[int(
                        class_id)].get(int(track_id), 0) + 1
            for face in face_this_frame:
                x1_face, y1_face, x2_face, y2_face = face
                for track in tracks:
                    x1, y1, x2, y2, track_id, confidence, class_id, _ = track
                    x1, y1, x2, y2 = int(x1 + xmin), int(y1 + ymin), int(
                        x2 + xmin), int(y2 + ymin)
                    if y1_face >= y1 and y2_face <= y2 and x1_face >= x1 and x2_face <= x2:
                        people_look_this_frame.append(track_id)
                        break
            if people_look_this_frame:
                total_people_look_camera.update(people_look_this_frame)
        cnt = [0, 0, 0]
        for vehicle in range(3):
            for id, value in numOfVehicle[vehicle].items():
                if value >= fps // 5:
                    cnt[vehicle] += 1

        progress = (frame_number / total_frames) * 100
        end_time = time.time()
        total_frame_time = end_time - start_time
        sum_time += total_frame_time
        cur_FPS = 1 / total_frame_time
        print(
            f"Processing frame {int(frame_number)} / {int(total_frames)} ({progress:.2f}%)")
        table_data = [
            ["Category", "4 wheeled vehicle",
                "2 wheeled vehicle", "People look camera", "Current FPS", "Time to track", "Time to predict"],
            ["Count", cnt[0], cnt[2], len(
                total_people_look_camera), f"{cur_FPS:.2f}", f"{time_tracker:.2f}", f"{time.time() - time_start_predict:.2f}"]
        ]

        # Định nghĩa chiều rộng của cột
        col_widths = [max(len(str(cell)) for cell in column)
                      for column in zip(*table_data)]

        # In các hàng đầu tiên (header)
        header = " | ".join(
            f"{cell:<{col_widths[i]}}" for i, cell in enumerate(table_data[0]))
        print(header)
        print("-" * (sum(col_widths) + 3 * (len(col_widths) - 1)))

        # In các hàng dữ liệu
        for row in table_data[1:]:
            print(" | ".join(
                f"{cell:<{col_widths[i]}}" for i, cell in enumerate(row)))
    print(f"Total time: {sum_time}")
    print(f"Average time to track: {sum_time_tracker / total_frames}")
    print(f"Average FPS: {total_frames / sum_time}")
    vid.release()
    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_video', type=int, default=10)
    parser.add_argument('--type_video', type=str, default='mp4')
    parser.add_argument('--name_model', type=str, default='new_md_5')
    parser.add_argument('--conf_thres', type=float, default=0.45)
    args = parser.parse_args()

    args.name_video = f"hota{args.num_video}"
    main(args)
