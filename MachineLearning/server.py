import asyncio
import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiohttp import web
from pyngrok import ngrok
class OpenCVVideoStreamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            raise RuntimeError("Nu s-a putut deschide camera video. Verifică conexiunea la cameră.")

    async def recv(self):
        frame = await asyncio.get_event_loop().run_in_executor(None, self._get_frame)
        return frame

    def _get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Eroare la capturarea cadrelor video.")
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

pcs = set()
async def offer(request):
    params = await request.json()
    pc = RTCPeerConnection()
    pcs.add(pc)
    video_track = None
    if cv2.VideoCapture(0).isOpened():
        video_track = OpenCVVideoStreamTrack()
        video_track.direction = "sendrecv"
        pc.addTrack(video_track)
    else:
        return web.Response(status=500, text="Eroare la inițializarea camerei. Camera nu este disponibilă.")

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        print(f"ICE connection state: {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    if "sdp" not in params or "type" not in params:
        return web.Response(status=400, text="Oferta este invalidă.")

    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.json_response(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )

async def cleanup(app):
    for pc in pcs:
        await pc.close()
    pcs.clear()

app = web.Application()
app.on_shutdown.append(cleanup)
app.router.add_post("/offer", offer)
public_url = ngrok.connect(8080)
print(f"Ngrok public URL: {public_url}")
web.run_app(app, host="0.0.0.0", port=8080)
