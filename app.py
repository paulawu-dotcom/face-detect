from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import requests
import base64
from PIL import Image
import io

app = Flask(__name__)

@app.route('/detect_face', methods=['POST'])
def detect_face():
    try:
        data = request.get_json()
        img_base64 = data.get('image')
        
        # 解碼 base64 圖片
        img_bytes = base64.b64decode(img_base64)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_array = np.array(img)
        
        # 偵測人臉
        face_locations = face_recognition.face_locations(img_array, model='hog')
        
        if not face_locations:
            return jsonify({'found': False, 'message': '找不到人臉'})
        
        # 取第一個人臉（top, right, bottom, left）
        top, right, bottom, left = face_locations[0]
        
        h, w = img_array.shape[:2]
        
        # 擴大範圍：頭頂多留 40%，兩側各留 20%，下方到肩膀多留 60%
        pad_top    = int((bottom - top) * 0.4)
        pad_side   = int((right - left) * 0.2)
        pad_bottom = int((bottom - top) * 0.6)
        
        x1 = max(0, left   - pad_side)
        y1 = max(0, top    - pad_top)
        x2 = min(w, right  + pad_side)
        y2 = min(h, bottom + pad_bottom)
        
        return jsonify({
            'found': True,
            'x': x1,
            'y': y1,
            'w': x2 - x1,
            'h': y2 - y1,
            'img_width':  w,
            'img_height': h
        })
        
    except Exception as e:
        return jsonify({'found': False, 'error': str(e)})

@app.route('/', methods=['GET'])
def health():
    return 'OK'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
