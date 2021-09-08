import jetson.inference
import jetson.utils
import cv2

app = Flask(__name__)
CORS(app, supports_credentials=True)

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.videoSource("rtsp://192.168.1.166:8554/unicast")      # '/dev/video0' for V4L2
display = jetson.utils.videoOutput("rtp://192.168.1.11:5000") # 'my_video.mp4' for file

while True:
    img = camera.Capture()
    detections = net.Detect(img)
    for detection in detections:
        print(detection)
    display.Render(img)
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
    imgOut = jetson.utils.cudaToNumpy(img)
