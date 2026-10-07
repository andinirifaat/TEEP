# code/pick_roi.py -- klik 4 titik ROI pada satu frame video
# urutan: kiri-dekat, kanan-dekat, kanan-jauh, kiri-jauh. Tekan Esc selesai.
import cv2, sys

cap = cv2.VideoCapture(sys.argv[1])
cap.set(cv2.CAP_PROP_POS_FRAMES, int(sys.argv[2]) if len(sys.argv) > 2 else 0)
ok, frame = cap.read()
pts = []

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))
        print(f"titik {len(pts)}: ({x}, {y})")

cv2.namedWindow("pilih ROI", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("pilih ROI", on_click)
while True:
    disp = frame.copy()
    for p in pts:
        cv2.circle(disp, p, 8, (0, 0, 255), -1)
    if len(pts) >= 2:
        cv2.polylines(disp, [__import__("numpy").array(pts)], len(pts) == 4,
                      (0, 255, 0), 3)
    cv2.imshow("pilih ROI", disp)
    if cv2.waitKey(30) == 27 or len(pts) == 4 and cv2.waitKey(500) == 27:
        break
print("\nTempel ke risk_dual_inference.py:")
print(f"ROI_POLYGON = np.array({[list(p) for p in pts]}, np.int32)")
cv2.destroyAllWindows()