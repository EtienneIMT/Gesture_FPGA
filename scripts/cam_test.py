# scripts/cam_test.py
import cv2

cap = cv2.VideoCapture(0)  # 0 = premier périphérique
if not cap.isOpened():
    print("Caméra inaccessible")
    exit(1)
ret, frame = cap.read()
print("Frame OK:", ret, "Shape:", None if not ret else frame.shape)
cv2.imshow("test", frame)
cv2.waitKey(2000)
cap.release()
cv2.destroyAllWindows()
