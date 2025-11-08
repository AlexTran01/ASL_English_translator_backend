import cv2
import torch
import numpy as np
import mediapipe as mp

# --------------------------
# 1. Load pretrained model
# --------------------------
model = torch.load("pretrained_model_for_WLASL2000.pt", map_location="cpu")
model.eval()

# --------------------------
# 2. Initialize Mediapipe Hands or Holistic
# --------------------------
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --------------------------
# 3. Helper to extract keypoints (skeleton)
# --------------------------
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

# --------------------------
# 4. Setup video capture
# --------------------------
cap = cv2.VideoCapture(0)
sequence = []
threshold = 0.7  # adjust for your model

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-30:]  # use last 30 frames

    # --------------------------
    # 5. Predict when enough frames collected
    # --------------------------
    if len(sequence) == 30:
        input_tensor = torch.tensor([sequence], dtype=torch.float32)
        with torch.no_grad():
            preds = model(input_tensor)
            pred_label = torch.argmax(preds, dim=1).item()
            confidence = torch.softmax(preds, dim=1)[0][pred_label].item()

        if confidence > threshold:
            cv2.putText(frame, f"Sign: {pred_label} ({confidence:.2f})", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # --------------------------
    # 6. Show frame
    # --------------------------
    cv2.imshow('WLASL Live Recognition', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
