from flask import Blueprint, request, jsonify, send_file
from flask_jwt_extended import jwt_required, get_jwt_identity
import base64
import io
import os
from datetime import datetime
import firebase_admin
from firebase_admin import firestore
import tempfile

from services.speech_service import SpeechService

speech_bp = Blueprint('speech', __name__)
speech_service = SpeechService()

@speech_bp.route('/transcribe', methods=['POST'])
@jwt_required()
def transcribe_audio():
    """Convert speech to text"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        # Get audio data (base64 encoded)
        audio_data = data.get('audio_data')
        room_id = data.get('room_id')
        language = data.get('language', 'en-US')
        
        if not audio_data:
            return jsonify({'error': 'No audio data provided'}), 400
        
        # Decode base64 audio
        try:
            if audio_data.startswith('data:audio'):
                audio_data = audio_data.split(',')[1]
            
            audio_bytes = base64.b64decode(audio_data)
            
        except Exception as e:
            return jsonify({'error': f'Invalid audio data: {str(e)}'}), 400
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
        
        try:
            # Transcribe audio
            transcription_result = speech_service.transcribe_audio(temp_audio_path, language)
            
            # Log transcription to Firestore
            if room_id and transcription_result.get('text'):
                db = firestore.client()
                transcription_log = {
                    'user_id': current_user_id,
                    'room_id': room_id,
                    'text': transcription_result.get('text'),
                    'confidence': transcription_result.get('confidence'),
                    'language': language,
                    'timestamp': datetime.utcnow()
                }
                db.collection('speech_transcriptions').add(transcription_log)
            
            return jsonify({
                'success': True,
                'text': transcription_result.get('text'),
                'confidence': transcription_result.get('confidence'),
                'language': language,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@speech_bp.route('/synthesize', methods=['POST'])
@jwt_required()
def synthesize_speech():
    """Convert text to speech"""
    try:
        data = request.get_json()
        text = data.get('text')
        language = data.get('language', 'en')
        voice = data.get('voice', 'default')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Generate speech
        audio_data = speech_service.synthesize_speech(text, language, voice)
        
        if audio_data:
            return jsonify({
                'success': True,
                'audio_data': base64.b64encode(audio_data).decode('utf-8'),
                'text': text,
                'language': language,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
        else:
            return jsonify({'error': 'Failed to synthesize speech'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@speech_bp.route('/text-to-asl', methods=['POST'])
@jwt_required()
def text_to_asl():
    """Convert text to ASL representation (GIF/avatar)"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        text = data.get('text')
        room_id = data.get('room_id')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Convert text to ASL representation
        asl_result = speech_service.text_to_asl(text)
        
        # Log conversion to Firestore
        if room_id:
            db = firestore.client()
            conversion_log = {
                'user_id': current_user_id,
                'room_id': room_id,
                'input_text': text,
                'asl_gestures': asl_result.get('gestures'),
                'gif_url': asl_result.get('gif_url'),
                'timestamp': datetime.utcnow()
            }
            db.collection('text_to_asl_conversions').add(conversion_log)
        
        return jsonify({
            'success': True,
            'input_text': text,
            'asl_gestures': asl_result.get('gestures'),
            'gif_url': asl_result.get('gif_url'),
            'avatar_data': asl_result.get('avatar_data'),
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@speech_bp.route('/stream-transcribe', methods=['POST'])
@jwt_required()
def stream_transcribe():
    """Real-time streaming transcription"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        audio_chunk = data.get('audio_chunk')  # Base64 encoded audio chunk
        room_id = data.get('room_id')
        chunk_timestamp = data.get('timestamp')
        is_final = data.get('is_final', False)
        
        if not audio_chunk:
            return jsonify({'error': 'No audio chunk provided'}), 400
        
        # Decode audio chunk
        try:
            if audio_chunk.startswith('data:audio'):
                audio_chunk = audio_chunk.split(',')[1]
            
            audio_bytes = base64.b64decode(audio_chunk)
            
        except Exception as e:
            return jsonify({'error': f'Invalid audio chunk: {str(e)}'}), 400
        
        # Process streaming transcription
        result = speech_service.process_streaming_audio(audio_bytes, is_final)
        
        # Return interim or final results
        if result.get('text'):
            # Log final transcriptions
            if is_final and room_id:
                db = firestore.client()
                transcription_log = {
                    'user_id': current_user_id,
                    'room_id': room_id,
                    'text': result.get('text'),
                    'confidence': result.get('confidence'),
                    'is_final': is_final,
                    'timestamp': datetime.utcnow(),
                    'chunk_timestamp': chunk_timestamp
                }
                db.collection('speech_transcriptions').add(transcription_log)
            
            return jsonify({
                'text': result.get('text'),
                'confidence': result.get('confidence'),
                'is_final': is_final,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
        else:
            return jsonify({
                'text': '',
                'is_final': is_final,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@speech_bp.route('/languages', methods=['GET'])
def get_supported_languages():
    """Get list of supported languages for speech recognition"""
    try:
        languages = speech_service.get_supported_languages()
        return jsonify({
            'languages': languages,
            'total_count': len(languages)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@speech_bp.route('/voices', methods=['GET'])
def get_available_voices():
    """Get list of available voices for speech synthesis"""
    try:
        voices = speech_service.get_available_voices()
        return jsonify({
            'voices': voices,
            'total_count': len(voices)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@speech_bp.route('/history', methods=['GET'])
@jwt_required()
def get_speech_history():
    """Get user's speech processing history"""
    try:
        current_user_id = get_jwt_identity()
        room_id = request.args.get('room_id')
        limit = int(request.args.get('limit', 50))
        
        db = firestore.client()
        query = db.collection('speech_transcriptions').where('user_id', '==', current_user_id)
        
        if room_id:
            query = query.where('room_id', '==', room_id)
        
        query = query.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit)
        
        transcriptions = []
        for doc in query.stream():
            transcription_data = doc.to_dict()
            transcription_data['id'] = doc.id
            transcription_data['timestamp'] = transcription_data['timestamp'].isoformat()
            transcriptions.append(transcription_data)
        
        return jsonify({
            'transcriptions': transcriptions,
            'total_count': len(transcriptions)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500 