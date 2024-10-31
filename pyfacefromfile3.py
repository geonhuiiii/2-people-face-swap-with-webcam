import insightface, cv2
from insightface.app import FaceAnalysis
from insightface.app.common import Face

cap = cv2.VideoCapture(0)
ret, frame = cap.read() 
app = FaceAnalysis(name="buffalo_l",allowed_modules=['detection','recognition'], providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))
swapper = insightface.model_zoo.get_model('./inswapper_128_fp16.onnx',providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
source = cv2.imread("./source.jpg")

if ret:
    try:
        myfaces = app.get(frame)
        sourcefaces = app.get(source)
        result = swapper.get(frame, myfaces[0], sourcefaces[0])
        cv2.imwrite("./changed.jpg", result) 
        cv2.imshow('cam', result)
        print("얼굴바꾸기 성공! changed.jpg 파일에 저장")
        cv2.waitKey()
    except Exception as E:
        cv2.imshow('cam', frame)
        print("얼굴바꾸기 실패...")
        print(E)
        cv2.waitKey()
cap.release()
cv2.destroyAllWindows()