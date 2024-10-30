import insightface
from insightface.app import FaceAnalysis
from insightface.app.common import Face
import cv2

app = FaceAnalysis(name="buffalo_l",allowed_modules=['detection','recognition'], providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))

cap = cv2.VideoCapture(0)

once = 0
swapper = insightface.model_zoo.get_model('./inswapper_128_fp16.onnx',providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])

while True:
    ret, frame = cap.read() 
    if not ret:
        break 
    if once:
        image = cv2.imread("./changed.jpg")
        cv2.imshow('image', image)

    cv2.imshow('cam', frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break 
    if k == ord('s'):
        try:
            myface = app.get(frame)
            firstface = Face(**myface[0])
            secondface = Face(**myface[1])
            firstface.embedding *= 1.5
            secondface.embedding *= 1.5


            result = swapper.get(frame, myface[0], secondface)
            result = swapper.get(result, myface[1], firstface)
            cv2.imwrite("./changed.jpg", result) 
            print("얼굴바꾸기 성공! changed.jpg 파일에 저장")
            once = True
        except Exception as E:
            print("얼굴바꾸기 실패...")
            print(E)

cap.release()
cv2.destroyAllWindows()