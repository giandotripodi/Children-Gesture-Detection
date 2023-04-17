import csv
import copy
import argparse
import itertools
import torch

import cv2 as cv
import numpy as np
import mediapipe as mp
from scipy.spatial import distance
from model import GestureClassifierEucl


def main():
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

    gesture_classifier_eucl = GestureClassifierEucl()

    with open('model/dataset_eucl/dataset_label.csv',
              encoding='utf-8-sig') as f:
        gesture_classifier_eucl_labels = csv.reader(f)
        gesture_classifier_eucl_labels = [
            row[0] for row in gesture_classifier_eucl_labels
        ]

    args = get_args()

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=4,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    video_path = "video/motherT0.MOV"
    video_name = video_path[6:]
    print(video_name)
    cap = cv.VideoCapture(video_path)

    MARGIN = 10
    mode = 0
    frame_count = 0
    fps = cap.get(cv.CAP_PROP_FPS)

    # with open('timeseries/' + video_name[:-4] + '_ts.csv', 'w', newline='') as csvfile:
    #     fieldnames = ['FRAME', 'TIME', 'TYPE']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()

    # file = open('timeseries/' + video_name + '_ts.txt', 'a', newline="")
    # file.write("FRAME,TIME,TYPE\n")
    # file.close()

    while cap.isOpened():
        ret, frame = cap.read()
        h, w, _ = frame.shape
        size = (w, h)
        print(size)
        break

    cap = cv.VideoCapture(video_path)

    with open('timeseries/' + video_name[:-4] + '.csv', 'w', newline='') as csvfile:
        fieldnames = ['FRAME', 'TIME', 'TYPE']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            time = round(float(frame_count) / fps, 2)

            image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image.flags.writeable = False
            result = yolo_model(image)
            image.flags.writeable = True

            # image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

            # Process Key (ESC: end) #
            key = cv.waitKey(10)
            if key == 27:  # ESC
                break

            number, mode = select_mode(key, mode)

            for (xmin, ymin, xmax, ymax, confidence, clas) in result.xyxy[0].tolist():
                debug_image = image[int(ymin) + MARGIN:int(ymax) + MARGIN, int(xmin) + MARGIN:int(xmax) + MARGIN:]
                results = hands.process(debug_image)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:

                        brect = calc_bounding_rect(debug_image, hand_landmarks)
                        landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                        pre_processed_landmark_list = pre_process_landmark_eucl(landmark_list)
                        logging_csv(number, mode, pre_processed_landmark_list)
                        hand_sign_id = gesture_classifier_eucl(pre_processed_landmark_list)
                        debug_image = draw_landmarks(debug_image, landmark_list)

                        # Pointing detected
                        if hand_sign_id == 1:
                            print(frame_count, time, "Pointing")
                            debug_image = draw_info_text(
                                debug_image,
                                brect,
                                gesture_classifier_eucl_labels[hand_sign_id]
                            )
                            row = {'FRAME': frame_count, 'TIME': time, 'TYPE': 'POINTING'}
                            writer.writerow(row)

                            # file = open('timeseries/' + video_name + '_ts.txt', 'a', newline="")
                            # file.write(str(frame_count) + ",")
                            # file.write(str(time) + ",")
                            # file.write("POINTING\n")
                            # file.close()

                        # No pointing detected
                        if hand_sign_id == 0:
                            print(frame_count, time, "No pointing")
                            debug_image = draw_info_text(
                                debug_image,
                                brect,
                                gesture_classifier_eucl_labels[hand_sign_id]
                            )

                        debug_image = draw_info(debug_image, mode, number)
                cv.imshow('Pointing Detection', debug_image)
        cap.release()
        cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Conversion into relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Conversion one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_landmark_eucl(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_point = temp_landmark_list[0]

    for i, j in enumerate(temp_landmark_list):
        p1 = np.array(base_point).reshape(1, -1)
        p2 = np.array(temp_landmark_list[i]).reshape(1, -1)
        d1 = distance.cdist(p1, p2, 'euclidean')
        temp_landmark_list[i] = d1.tolist()

    temp_landmark_list = np.reshape(temp_landmark_list, 21)

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/dataset/dataset.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])

    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 3)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 1)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 3)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 1)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 3)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 1)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 3)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 1)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 3)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 1)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 3)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 1)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 3)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 1)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 3)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 1)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 3)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 1)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 3)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 1)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 3)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 1)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 3)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 1)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 3)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 1)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 3)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 1)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 3)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 1)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 3)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 1)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 3)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 1)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 3)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 1)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 3)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 1)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 3)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 1)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 3)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 1)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:
            cv.circle(image, (landmark[0], landmark[1]), 2, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
        if index == 1:
            cv.circle(image, (landmark[0], landmark[1]), 2, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
        if index == 2:
            cv.circle(image, (landmark[0], landmark[1]), 2, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
        if index == 3:
            cv.circle(image, (landmark[0], landmark[1]), 2, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
        if index == 4:
            cv.circle(image, (landmark[0], landmark[1]), 2, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
        if index == 5:
            cv.circle(image, (landmark[0], landmark[1]), 2, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
        if index == 6:
            cv.circle(image, (landmark[0], landmark[1]), 2, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
        if index == 7:
            cv.circle(image, (landmark[0], landmark[1]), 2, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
        if index == 8:
            cv.circle(image, (landmark[0], landmark[1]), 2, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
        if index == 9:
            cv.circle(image, (landmark[0], landmark[1]), 2, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
        if index == 10:
            cv.circle(image, (landmark[0], landmark[1]), 2, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
        if index == 11:
            cv.circle(image, (landmark[0], landmark[1]), 2, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
        if index == 12:
            cv.circle(image, (landmark[0], landmark[1]), 2, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
        if index == 13:
            cv.circle(image, (landmark[0], landmark[1]), 2, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
        if index == 14:  #
            cv.circle(image, (landmark[0], landmark[1]), 2, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
        if index == 15:
            cv.circle(image, (landmark[0], landmark[1]), 2, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
        if index == 16:
            cv.circle(image, (landmark[0], landmark[1]), 2, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
        if index == 17:
            cv.circle(image, (landmark[0], landmark[1]), 2, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
        if index == 18:
            cv.circle(image, (landmark[0], landmark[1]), 2, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
        if index == 19:
            cv.circle(image, (landmark[0], landmark[1]), 2, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
        if index == 20:
            cv.circle(image, (landmark[0], landmark[1]), 2, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)
    if hand_sign_text != "":
        cv.putText(image, hand_sign_text, (brect[0] + 5, brect[1] - 4),
                   cv.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image


def draw_info(image, mode, number):
    if mode == 1:
        cv.putText(image, "RECORDING GESTURE MODE, press n to cancel", (10, 150),
                   cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "LABEL RECORDED:" + str(number), (10, 170),
                       cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 26), 1,
                       cv.LINE_AA)
    return image


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
