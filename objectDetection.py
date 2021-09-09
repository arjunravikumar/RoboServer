import jetson.inference
import jetson.utils
import cv2

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.videoSource("rtsp://192.168.1.166:8554/unicast")
display = jetson.utils.videoOutput("rtp://127.0.0.1:8554/test","--headless")

while True:
    img = camera.Capture()
    detections = net.Detect(img)
    for detection in detections:
        print(detection)
    display.Render(img)
