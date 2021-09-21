import websocket
import time
import _thread

class RoboControls:
    ws = None
    robotIsMobile = False
    def startWS(self):
        websocket.enableTrace(True)
        self.ws = websocket.WebSocketApp("ws://192.168.1.166:8888",
                                  on_message = self.on_message,
                                  on_error = self.on_error,
                                  on_close = self.on_close)
        self.ws.on_open = self.on_open
        self.ws.run_forever()

    def on_message(self,ws, message):
        print(message)

    def on_error(self,ws, error):
        print(error)

    def on_close(self,ws,arg1=None,arg2=None):
        print("### closed ###")

    def closeWS(self):
        self.ws.close()

    def on_open(self,ws):
        def run(self):
            self.ws.send("tumbler:wakeup")
            print("Connected")
        _thread.start_new_thread(run, (self))

    def move(self,direction,speed=100):
        def run(self):
            print(direction,self.robotIsMobile)
            if(self.robotIsMobile == False):
                self.robotIsMobile = True
                self.ws.send(direction)
                print("Done Sending")
        _thread.start_new_thread(run, (self))

    def stopMovement(self):
        def run(self):
            print("stop",self.robotIsMobile)
            if(self.robotIsMobile == True):
                self.robotIsMobile = False
                self.ws.send("DS")
                print("Done Sending")
            _thread.start_new_thread(run, (self))