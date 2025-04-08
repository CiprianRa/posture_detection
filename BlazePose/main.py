import cv2
import mediapipe as mp
import time

# === Parametri pentru netezire ===
alpha = 0.2
delta_threshold = 0.005  # prag pentru ignorare zgomot
shoulder_z_smoothed = None
hip_z_smoothed = None
shoulder_y_smoothed = None
hip_y_smoothed = None

# === Postură și temporizare ===
current_posture = None
posture_timer_start = None
required_duration = 0.8  # secunde pentru confirmare postură nouă

# === Funcție pentru filtrare adaptivă ===
def smooth_with_threshold(current_value, previous_smoothed, alpha, delta):
    if abs(current_value - previous_smoothed) < delta:
        return previous_smoothed
    return alpha * current_value + (1 - alpha) * previous_smoothed

# === Inițializare MediaPipe ===
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8) as pose:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Media z pentru umeri și șolduri
            shoulder_avg_z = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z +
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z) / 2
            hip_avg_z = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].z +
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z) / 2
            shoulder_avg_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y +
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
            hip_avg_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y +
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2

            # Aplicăm filtrare adaptivă (EMA + threshold)
            if shoulder_z_smoothed is None:
                shoulder_z_smoothed = shoulder_avg_z
                hip_z_smoothed = hip_avg_z
                shoulder_y_smoothed = shoulder_avg_y
                hip_y_smoothed = hip_avg_y
            else:
                shoulder_z_smoothed = smooth_with_threshold(shoulder_avg_z, shoulder_z_smoothed, alpha, delta_threshold)
                hip_z_smoothed = smooth_with_threshold(hip_avg_z, hip_z_smoothed, alpha, delta_threshold)
                shoulder_y_smoothed = smooth_with_threshold(shoulder_avg_y, shoulder_y_smoothed, alpha, delta_threshold)
                hip_y_smoothed = smooth_with_threshold(hip_avg_y, hip_y_smoothed, alpha, delta_threshold)

            # Diferența z între șolduri și umeri
            z_diff = hip_z_smoothed - shoulder_z_smoothed
            y_diff = hip_y_smoothed - shoulder_y_smoothed

            # Propunem o nouă postură
            if y_diff < 0.5:
                if z_diff > 0.14:
                    new_posture = "APLECAT"
                elif z_diff < 0.12:
                    new_posture = "DREPT"
                else:
                    new_posture = current_posture
            else:
                new_posture = "DREPT"

            # Aplicăm temporizare pentru schimbare postură
            if new_posture != current_posture:
                if posture_timer_start is None:
                    posture_timer_start = time.time()
                elif time.time() - posture_timer_start >= required_duration:
                    current_posture = new_posture
                    posture_timer_start = None
            else:
                posture_timer_start = None

            # Culoare în funcție de postură
            color = (0, 255, 0) if current_posture == "DREPT" else (0, 0, 255)

            # Afișare rezultate
            cv2.putText(image, f"Z_DIFF: {z_diff:.4f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            cv2.putText(image, f"POSTURA: {current_posture}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

        # Fereastră video
        cv2.imshow('Postura Stabilizată (Media Z umeri/șolduri)', image)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
