import time
import asyncio
import websockets


class RoboControls:
    websocket = None
    robotIsMobile = False

    async def startWebSocket(self):
        async with websockets.connect('ws://192.168.1.166:8888') as self.websocket:
            await self.websocket.send("tumbler:wakeup")
            msgRecv = await self.websocket.recv()
            self.messageRecieved(msgRecv)

    def messageRecieved(self,msgRecv):
        print("message from server: ",msgRecv)

    async def move(self,direction,speed=100):
        if(self.robotIsMobile == False):
            self.robotIsMobile = True
            await self.websocket.send(direction)

    async def stopMovement(self):
        if(self.robotIsMobile == True):
            self.robotIsMobile = False
            await self.websocket.send("DS")

    def __init__(self):
        asyncio.get_event_loop().run_until_complete(self.startWebSocket())