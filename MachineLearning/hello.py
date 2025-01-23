import subprocess 
import cv2

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

command = ['ffmpeg',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s','640x480',
            '-i','-',
            '-ar', '44100',
            '-ac', '2',
            '-acodec', 'pcm_s16le',
            '-f', 's16le',
            '-ac', '2',
            '-i','/dev/zero',   
            '-acodec','aac',
            '-ab','128k',
            '-strict','experimental',
            '-vcodec','h264',
            '-pix_fmt','yuv420p',
            '-g', '50',
            '-vb','2500k',
            '-profile:v', 'baseline',
            '-preset', 'ultrafast',
            '-r', '30',
            '-f', 'flv', 
            'rtmp://a.rtmp.youtube.com/live2/d4jp-wysv-7e8q-67sp-3efu']


pipe = subprocess.Popen(command, stdin=subprocess.PIPE)
while True:
    ret, frame = cap.read()

    pipe.stdin.write(frame.tostring())

pipe.kill()
cap.release()
