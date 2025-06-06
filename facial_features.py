import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

# Load image
image = cv2.imread(r"human_faces\1 (4).jpeg")
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process
results = face_mesh.process(rgb_image)

# Draw landmarks
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

cv2.imshow("Face Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
