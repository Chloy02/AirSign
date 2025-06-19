from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
from flask_jwt_extended import JWTManager
import os
from dotenv import load_dotenv

# Import routes
from api.auth import auth_bp
from api.asl import asl_bp
from api.speech import speech_bp
from api.meeting import meeting_bp
from api.user import user_bp

# Import services
from services.firebase_service import initialize_firebase
from services.asl_service import ASLService
from services.speech_service import SpeechService

# Load environment variables
load_dotenv()

def create_app():
    """Application factory pattern for Flask app"""
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
    app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'jwt-secret-key-change-in-production')
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = False  # For development
    
    # Initialize extensions
    CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])
    jwt = JWTManager(app)
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    
    # Initialize Firebase
    initialize_firebase()
    
    # Initialize services
    asl_service = ASLService()
    speech_service = SpeechService()
    
    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(asl_bp, url_prefix='/api/asl')
    app.register_blueprint(speech_bp, url_prefix='/api/speech')
    app.register_blueprint(meeting_bp, url_prefix='/api/meeting')
    app.register_blueprint(user_bp, url_prefix='/api/user')
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Resource not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500
    
    # Health check endpoint
    @app.route('/api/health')
    def health_check():
        return jsonify({
            'status': 'healthy',
            'service': 'AirSign Backend',
            'version': '1.0.0'
        })
    
    # Socket.IO event handlers
    @socketio.on('connect')
    def handle_connect():
        print(f'Client connected: {request.sid}')
        emit('connected', {'data': 'Connected to AirSign server'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        print(f'Client disconnected: {request.sid}')
    
    @socketio.on('join_meeting')
    def handle_join_meeting(data):
        room = data.get('room_id')
        user_id = data.get('user_id')
        join_room(room)
        emit('user_joined', {
            'user_id': user_id,
            'room_id': room
        }, room=room)
    
    @socketio.on('leave_meeting')
    def handle_leave_meeting(data):
        room = data.get('room_id')
        user_id = data.get('user_id')
        leave_room(room)
        emit('user_left', {
            'user_id': user_id,
            'room_id': room
        }, room=room)
    
    @socketio.on('asl_gesture')
    def handle_asl_gesture(data):
        room = data.get('room_id')
        gesture_data = data.get('gesture_data')
        user_id = data.get('user_id')
        
        # Process ASL gesture
        result = asl_service.process_gesture(gesture_data)
        
        emit('gesture_recognized', {
            'user_id': user_id,
            'gesture': result,
            'timestamp': data.get('timestamp')
        }, room=room)
    
    @socketio.on('speech_input')
    def handle_speech_input(data):
        room = data.get('room_id')
        audio_data = data.get('audio_data')
        user_id = data.get('user_id')
        
        # Process speech input
        result = speech_service.process_speech(audio_data)
        
        emit('speech_processed', {
            'user_id': user_id,
            'text': result.get('text'),
            'asl_response': result.get('asl_response'),
            'timestamp': data.get('timestamp')
        }, room=room)
    
    return app, socketio

if __name__ == '__main__':
    app, socketio = create_app()
    socketio.run(app, host='0.0.0.0', port=5000, debug=True) 