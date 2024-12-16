import firebase_admin
from firebase_admin import credentials, firestore, storage
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import os
from datetime import datetime
import pytz
import uuid
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
service_account_path = os.path.join(BASE_DIR, 'megaanunt-firebase-adminsdk-4w8vs-139038af14.json')
cred = credentials.Certificate(service_account_path)
firebase_admin.initialize_app(cred, {
    'storageBucket': 'megaanunt.appspot.com',
})

db = firestore.client()
bucket = storage.bucket()

class UploadPredictionView(APIView):
    def post(self, request, *args, **kwargs):  
        file = request.FILES.get('file')
        classification = request.data.get('classification')
        detection_type = request.data.get('detection_type')
        confidence = request.data.get('confidence')
        auth_token = request.data.get('auth_token') 

        if not file or not file.name.endswith(('.wav', '.mp3', '.mp4', '.avi')):
            return Response({"error": "Please upload a valid .wav or video file."}, status=status.HTTP_400_BAD_REQUEST)

        token = uuid.uuid4().hex
        blob = None
        if (detection_type == "Video"):
            blob = bucket.blob(f'detections/{classification}_{token}.avi')
        else:
            blob = bucket.blob(f'detections/{classification}_{token}.wav')
        blob.upload_from_file(file, content_type='audio/wav' if file.name.endswith('.wav') else 'video/avi')

        blob.make_public()
        file_url = blob.public_url
        print(file_url)

        timezone = pytz.timezone('Europe/Chisinau')
        detection_time = datetime.now(timezone).timestamp()

        detection_data = {
            'classification': classification,
            'detection_type': detection_type,
            'confidence': int(confidence),
            'token': auth_token,
            'link': file_url,
            'detection_time': int(detection_time)
        }
        db.collection('alerts').add(detection_data)

        return Response({
            "message": "Detection data uploaded successfully",
            "data": detection_data
        }, status=status.HTTP_201_CREATED)

       
