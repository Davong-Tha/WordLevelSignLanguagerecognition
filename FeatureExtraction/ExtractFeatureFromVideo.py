import csv
import os

import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def extract_keypoints(results, hand=True, pose=True, face=False):
    pose_feature = np.array([[res.x, res.y, res.z, res.visibility] for res in
                             results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face_feature = np.array([[res.x, res.y, res.z] for res in
                             results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(
        468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)

    keypoints_array = []
    if pose:
        keypoints_array.append(pose_feature)
    if face:
        keypoints_array.append(face_feature)
    if hand:
        keypoints_array.append(lh)
        keypoints_array.append(rh)
    return np.concatenate(keypoints_array)


def draw_styled_landmarks(image, results, hand=True, pose=True, face=False):
    # Draw face connections
    if face:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                  )
    # Draw pose connections
    if pose:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                  )
    if hand:
        # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                  )
        # Draw right hand connections
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )


if __name__ == '__main__':
    vid_dir = '../dataset/lsa64_raw/video'
    for i, vid in enumerate(os.listdir(vid_dir)):
        if i < 597:
            continue
        print(str(i) + ' ' +vid)
        cap = cv2.VideoCapture(os.path.join(vid_dir, vid))
        frame_list = []
        landmark1 = []
        landmark2 = []
        # result = cv2.VideoWriter(f'example{i}.avi',
        #                          cv2.VideoWriter_fourcc(*'MJPG'),
        #                          10, (int(cap.get(3)), int(cap.get(4))))
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                # Read feed
                success, OriginalFrame = cap.read()
                if not success:
                    break


                while success:
                    frame_list.append(OriginalFrame)
                    success, OriginalFrame = cap.read()
            cap.release()

            for frame_num, OriginalFrame in enumerate(frame_list):
                FlippedFrame = cv2.flip(OriginalFrame, 1)
                OriginalFrame, result1 = mediapipe_detection(OriginalFrame, holistic)
                FlippedFrame, result2 = mediapipe_detection(FlippedFrame, holistic)

                OriginalFrameFeature = extract_keypoints(result1)
                FlippedFrameFeature = extract_keypoints(result2)
                landmark1.append(OriginalFrameFeature)
                landmark2.append(FlippedFrameFeature)
                # Draw landmarks
                # draw_styled_landmarks(OriginalFrame, result1)
                # draw_styled_landmarks(FlippedFrame, result2)
                # cv2.putText(OriginalFrame, 'frame: '+str(frame_num), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (36, 255, 12), 5)
                # cv2.putText(FlippedFrame, str(frame_num), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                # result.write(OriginalFrame)
                # cv2.imshow("original", OriginalFrame)

                # cv2.imshow("flipped", FlippedFrame)

                if cv2.waitKey(5) & 0xFF == 27:
                    break
        # result.release()
        extracted_path = f"../dataset/lsa64_raw/extracted/{vid}.csv"
        os.makedirs(os.path.dirname(extracted_path), exist_ok=True)
        with open(extracted_path, "w+", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(landmark1)

        extracted_path = f"../dataset/lsa64_raw/extracted/{vid}-flipped.csv"
        os.makedirs(os.path.dirname(extracted_path), exist_ok=True)
        with open(extracted_path, "w+", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(landmark2)


