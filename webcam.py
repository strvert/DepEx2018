import cv2


class Webcam:
    def __init__(self):
        pass

    def captureCam(self, mirror=True, size=None):
        cap = cv2.VideoCapture(0)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        pre_frame = cap.read
        while True:
            ret, frame = cap.read()
            cv2.imshow('capture', frame)

            k = cv2.waitKey(1)
            if k == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
