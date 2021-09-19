import websocket
import time

class RoboControls:
    ws = None
    robotIsMobile = False
    def __init__(self):
        websocket.enableTrace(True)
        self.ws = websocket.WebSocketApp("192.168.1.166:8888",
                                  on_message = self.on_message,
                                  on_error = self.on_error,
                                  on_close = self.on_close)
        self.ws.on_open = self.on_open
        self.ws.run_forever()

    def on_message(self,ws, message):
        print(message)

    def on_error(self,ws, error):
        print(error)

    def on_close(self,ws):
        print("### closed ###")

    def closeWS(self):
        self.ws.close()

    def on_open(self,ws):
        self.ws.send("tumbler:wakeup")

    def move(direction,speed=100):
        if(robotIsMobile == False):
            robotIsMobile = True
            self.ws.send(direction)

    def stopMovement():
        if(robotIsMobile == True):
            robotIsMobile = False
            self.ws.send("DS")