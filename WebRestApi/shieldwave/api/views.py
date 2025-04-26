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
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
service_account_path = os.path.join(BASE_DIR, 'megaanunt-firebase-adminsdk-4w8vs-139038af14.json')
cred = credentials.Certificate(service_account_path)
firebase_admin.initialize_app(cred, {
    'storageBucket': 'megaanunt.appspot.com',
})

db = firestore.client()
bucket = storage.bucket()


class CheckUserExists(APIView):
    def get(self, request, *args, **kwargs):
        auth_token = request.GET.get("auth_token")

        users_ref = db.collection("usersDB")
        query = users_ref.where("token", "==", auth_token).get()
        if not query:
            return Response({"state": "NoUser"}, status=200)
        
        return Response({"state": "YesUser"}, status=200)

class InitialCleanup(APIView):
    def get(self, request, *args, **kwargs):
        auth_token = request.GET.get("auth_token")

        users_ref = db.collection("usersDB")
        query = users_ref.where("token", "==", auth_token).get()

        if not query:
            return Response({"error": "No matching user found"}, status=404)

        user_doc = query[0]
        user_id = user_doc.id
        user_data = user_doc.to_dict()

        if (user_data.get("deactivateSystem") == "yes" or user_data.get("deactivateCam") == "yes" or 
            user_data.get("deactivateMic") == "yes" or user_data.get("activate") == "yes"):
            
            users_ref.document(user_id).update({
                "deactivateSystem": "no",
                "deactivateCam": "no",
                "deactivateMic": "no",
                "activate": "no"
            })

        return Response("User fields reset successfully!", status=200)



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
            blob = bucket.blob(f'detections/{token}.avi')
        else:
            blob = bucket.blob(f'detections/{token}.mp3')
        blob.upload_from_file(file, content_type='audio/mp3' if file.name.endswith('.mp3') else 'video/avi')

        blob.make_public()
        file_url = blob.public_url
        print(file_url)

        timezone = pytz.timezone('Europe/Chisinau')
        detection_time = datetime.now(timezone).timestamp()

        detection_data = {
            'classification': classification,
            'detection_type': detection_type,
            'confidence': int(confidence),
            'link': file_url,
            'detection_time': int(detection_time)
        }

        alerts_ref = db.collection('alerts')
        query = alerts_ref.where("user_token", "==", auth_token).get()
        if not query:
            db.collection('alerts').add({
                'user_token': auth_token,
                'alert_list': [detection_data]
            })
        else:
            doc_ref = query[0].reference
            doc_data = query[0].to_dict()
            
            existing_alerts = doc_data.get('alert_list', [])
            existing_alerts.append(detection_data)
            
            doc_ref.update({
                'alert_list': existing_alerts
            })


        return Response({
            "message": "Detection data uploaded successfully"
        }, status=status.HTTP_201_CREATED)




class GetUserStatusInfo(APIView):
    def get(self, request, *args, **kwargs):
        auth_token = request.GET.get("auth_token")
        print(f"Received token: {auth_token}")

        users_ref = db.collection("usersDB")
        query = users_ref.where("token", "==", auth_token).get()

        if not query:
            return Response({"error": "No matching user found"}, status=404)

        user_doc = query[0]
        user_data = user_doc.to_dict()
        document_id = user_doc.id

        status_info = {
            "deactivateSystem": user_data.get("deactivateSystem"),
            "deactivateCam": user_data.get("deactivateCam"),
            "deactivateMic": user_data.get("deactivateMic"),
            "activate": user_data.get("activate")
        }

        print(f"User status info: {status_info}")

        # Funcție pentru resetare
        def delayed_reset():
            time.sleep(27)
            users_ref.document(document_id).update({
                "deactivateSystem": "no",
                "deactivateCam": "no",
                "deactivateMic": "no",
                "activate": "no"
            })

        if any(value == "yes" for value in status_info.values()):
            threading.Thread(target=delayed_reset, daemon=True).start()

        return Response(status_info, status=200)


class ActivityInfo(APIView):
    def get(self, request, *args, **kwargs):
        auth_token = request.GET.get("auth_token")
        state = request.GET.get('deactivate_variable')
        state = (state == "True")
        users = db.collection("usersDB")
        query = users.where("token", "==", auth_token).get()
        if not query:
            return Response({"error": "No matching user found"}, status=404)

        document_id = query[0].id

        if (state == True):
            users.document(document_id).update({
                "state": "inactive",
                "last_active": int(time.time())
            })
        elif(state == False):
            users.document(document_id).update({
                "state": "active",
                "last_active": int(time.time())
            })
        
        return Response("Updated", status=200)