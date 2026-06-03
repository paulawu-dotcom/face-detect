from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from PIL import Image
import io

app = Flask(__name__)

# 載入 OpenCV 內建的人臉偵測模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/detect_face', methods=['POST'])
def detect_face():
    try:
        data = request.get_json()
        img_base64 = data.get('image')
        img_bytes = base64.b64decode(img_base64)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

        if len(faces) == 0:
            return jsonify({'found': False, 'message': '找不到人臉'})

        x, y, w, h = faces[0]
        H, W = img_array.shape[:2]

        pad_top    = int(h * 0.5)
        pad_side   = int(w * 0.3)
        pad_bottom = int(h * 0.8)

        x1 = max(0, x - pad_side)
        y1 = max(0, y - pad_top)
        x2 = min(W, x + w + pad_side)
        y2 = min(H, y + h + pad_bottom)

        return jsonify({
            'found': True,
            'x': int(x1), 'y': int(y1),
            'w': int(x2 - x1), 'h': int(y2 - y1),
            'img_width': W, 'img_height': H
        })
    except Exception as e:
        return jsonify({'found': False, 'error': str(e)})

@app.route('/', methods=['GET'])
def health():
    return 'OK'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
