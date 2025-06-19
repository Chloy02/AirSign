from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
import base64
import cv2
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime
import firebase_admin
from firebase_admin import firestore

from services.asl_service import ASLService

asl_bp = Blueprint('asl', __name__)
asl_service = ASLService()

@asl_bp.route('/detect', methods=['POST'])
@jwt_required()
def detect_gesture():
    """Detect ASL gesture from image data"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        # Get image data (base64 encoded)
        image_data = data.get('image_data')
        room_id = data.get('room_id')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        try:
            # Remove data URL prefix if present
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            return jsonify({'error': f'Invalid image data: {str(e)}'}), 400
        
        # Process gesture detection
        result = asl_service.detect_gesture(opencv_image)
        
        # Log detection to Firestore
        if room_id:
            db = firestore.client()
            detection_log = {
                'user_id': current_user_id,
                'room_id': room_id,
                'gesture': result.get('gesture'),
                'confidence': result.get('confidence'),
                'timestamp': datetime.utcnow(),
                'bbox': result.get('bbox')
            }
            db.collection('asl_detections').add(detection_log)
        
        return jsonify({
            'success': True,
            'gesture': result.get('gesture'),
            'confidence': result.get('confidence'),
            'bbox': result.get('bbox'),
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@asl_bp.route('/stream', methods=['POST'])
@jwt_required()
def process_gesture_stream():
    """Process continuous gesture stream for real-time detection"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        frame_data = data.get('frame_data')  # Base64 encoded frame
        room_id = data.get('room_id')
        frame_timestamp = data.get('timestamp')
        
        if not frame_data:
            return jsonify({'error': 'No frame data provided'}), 400
        
        # Decode frame
        try:
            if frame_data.startswith('data:image'):
                frame_data = frame_data.split(',')[1]
            
            frame_bytes = base64.b64decode(frame_data)
            frame = Image.open(io.BytesIO(frame_bytes))
            opencv_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            return jsonify({'error': f'Invalid frame data: {str(e)}'}), 400
        
        # Process frame for gesture detection
        result = asl_service.process_frame(opencv_frame)
        
        # Only return results if gesture is detected with sufficient confidence
        if result.get('gesture') and result.get('confidence', 0) > 0.7:
            # Log high-confidence detections
            if room_id:
                db = firestore.client()
                detection_log = {
                    'user_id': current_user_id,
                    'room_id': room_id,
                    'gesture': result.get('gesture'),
                    'confidence': result.get('confidence'),
                    'timestamp': datetime.utcnow(),
                    'frame_timestamp': frame_timestamp,
                    'bbox': result.get('bbox')
                }
                db.collection('asl_detections').add(detection_log)
            
            return jsonify({
                'gesture_detected': True,
                'gesture': result.get('gesture'),
                'confidence': result.get('confidence'),
                'bbox': result.get('bbox'),
                'timestamp': datetime.utcnow().isoformat()
            }), 200
        else:
            return jsonify({
                'gesture_detected': False,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@asl_bp.route('/gestures', methods=['GET'])
def get_available_gestures():
    """Get list of available ASL gestures"""
    try:
        gestures = asl_service.get_available_gestures()
        return jsonify({
            'gestures': gestures,
            'total_count': len(gestures)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@asl_bp.route('/model/status', methods=['GET'])
def get_model_status():
    """Get YOLOv8 model status and information"""
    try:
        status = asl_service.get_model_status()
        return jsonify(status), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@asl_bp.route('/model/reload', methods=['POST'])
@jwt_required()
def reload_model():
    """Reload the YOLOv8 model (admin function)"""
    try:
        # Check if user is admin (you might want to add admin role checking)
        current_user_id = get_jwt_identity()
        
        # Reload model
        success = asl_service.reload_model()
        
        if success:
            return jsonify({'message': 'Model reloaded successfully'}), 200
        else:
            return jsonify({'error': 'Failed to reload model'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@asl_bp.route('/history', methods=['GET'])
@jwt_required()
def get_detection_history():
    """Get user's ASL detection history"""
    try:
        current_user_id = get_jwt_identity()
        room_id = request.args.get('room_id')
        limit = int(request.args.get('limit', 50))
        
        db = firestore.client()
        query = db.collection('asl_detections').where('user_id', '==', current_user_id)
        
        if room_id:
            query = query.where('room_id', '==', room_id)
        
        query = query.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit)
        
        detections = []
        for doc in query.stream():
            detection_data = doc.to_dict()
            detection_data['id'] = doc.id
            detection_data['timestamp'] = detection_data['timestamp'].isoformat()
            detections.append(detection_data)
        
        return jsonify({
            'detections': detections,
            'total_count': len(detections)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500 