from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime
import firebase_admin
from firebase_admin import firestore, auth

user_bp = Blueprint('user', __name__)

@user_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_user_profile():
    """Get current user's complete profile"""
    try:
        current_user_id = get_jwt_identity()
        
        db = firestore.client()
        user_doc = db.collection('users').document(current_user_id).get()
        
        if not user_doc.exists:
            return jsonify({'error': 'User not found'}), 404
        
        user_data = user_doc.to_dict()
        
        # Get additional stats
        stats = get_user_stats(current_user_id)
        
        return jsonify({
            'user': {
                'uid': user_data.get('uid'),
                'email': user_data.get('email'),
                'name': user_data.get('name'),
                'user_type': user_data.get('user_type'),
                'preferences': user_data.get('preferences', {}),
                'created_at': user_data.get('created_at').isoformat() if user_data.get('created_at') else None,
                'last_login': user_data.get('last_login').isoformat() if user_data.get('last_login') else None,
                'stats': stats
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@user_bp.route('/profile', methods=['PUT'])
@jwt_required()
def update_user_profile():
    """Update user profile information"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        db = firestore.client()
        update_data = {}
        
        # Validate and update fields
        if 'name' in data:
            update_data['name'] = data['name']
        
        if 'user_type' in data:
            valid_types = ['standard', 'hearing_impaired', 'speech_impaired']
            if data['user_type'] in valid_types:
                update_data['user_type'] = data['user_type']
            else:
                return jsonify({'error': 'Invalid user type'}), 400
        
        if 'preferences' in data:
            current_prefs = db.collection('users').document(current_user_id).get().to_dict().get('preferences', {})
            current_prefs.update(data['preferences'])
            update_data['preferences'] = current_prefs
        
        if update_data:
            update_data['updated_at'] = datetime.utcnow()
            db.collection('users').document(current_user_id).update(update_data)
        
        return jsonify({
            'message': 'Profile updated successfully',
            'updated_fields': list(update_data.keys())
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@user_bp.route('/preferences', methods=['GET'])
@jwt_required()
def get_user_preferences():
    """Get user's accessibility preferences"""
    try:
        current_user_id = get_jwt_identity()
        
        db = firestore.client()
        user_doc = db.collection('users').document(current_user_id).get()
        
        if not user_doc.exists:
            return jsonify({'error': 'User not found'}), 404
        
        user_data = user_doc.to_dict()
        preferences = user_data.get('preferences', {})
        
        return jsonify({
            'preferences': preferences
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@user_bp.route('/preferences', methods=['PUT'])
@jwt_required()
def update_user_preferences():
    """Update user's accessibility preferences"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No preferences provided'}), 400
        
        db = firestore.client()
        user_doc = db.collection('users').document(current_user_id).get()
        
        if not user_doc.exists:
            return jsonify({'error': 'User not found'}), 404
        
        user_data = user_doc.to_dict()
        current_preferences = user_data.get('preferences', {})
        
        # Update preferences
        current_preferences.update(data)
        
        # Validate preference values
        valid_preferences = {
            'asl_enabled': bool,
            'speech_enabled': bool,
            'subtitle_enabled': bool,
            'voice_speed': (int, float),
            'subtitle_size': str,
            'subtitle_color': str,
            'asl_sensitivity': (int, float),
            'auto_caption': bool
        }
        
        for key, value in current_preferences.items():
            if key in valid_preferences:
                expected_type = valid_preferences[key]
                if not isinstance(value, expected_type):
                    return jsonify({'error': f'Invalid type for {key}'}), 400
        
        # Update in database
        db.collection('users').document(current_user_id).update({
            'preferences': current_preferences,
            'updated_at': datetime.utcnow()
        })
        
        return jsonify({
            'message': 'Preferences updated successfully',
            'preferences': current_preferences
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@user_bp.route('/stats', methods=['GET'])
@jwt_required()
def get_user_stats():
    """Get user's usage statistics"""
    try:
        current_user_id = get_jwt_identity()
        stats = get_user_stats(current_user_id)
        
        return jsonify({
            'stats': stats
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@user_bp.route('/meetings/history', methods=['GET'])
@jwt_required()
def get_meeting_history():
    """Get user's meeting history"""
    try:
        current_user_id = get_jwt_identity()
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        
        db = firestore.client()
        
        # Get meetings where user participated
        meetings_query = db.collection('meetings')\
            .where('participants', 'array_contains_any', [{'user_id': current_user_id}])\
            .order_by('created_at', direction=firestore.Query.DESCENDING)\
            .limit(limit).offset(offset)
        
        meetings = []
        for doc in meetings_query.stream():
            meeting_data = doc.to_dict()
            meeting_data['id'] = doc.id
            
            # Find user's role in this meeting
            participants = meeting_data.get('participants', [])
            user_participant = next(
                (p for p in participants if p['user_id'] == current_user_id), None
            )
            meeting_data['user_role'] = 'host' if user_participant and user_participant.get('is_host') else 'participant'
            
            meetings.append(meeting_data)
        
        return jsonify({
            'meetings': meetings,
            'total_count': len(meetings),
            'has_more': len(meetings) == limit
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@user_bp.route('/activity', methods=['GET'])
@jwt_required()
def get_user_activity():
    """Get user's recent activity"""
    try:
        current_user_id = get_jwt_identity()
        days = int(request.args.get('days', 7))
        
        db = firestore.client()
        
        # Get recent ASL detections
        asl_detections = db.collection('asl_detections')\
            .where('user_id', '==', current_user_id)\
            .order_by('timestamp', direction=firestore.Query.DESCENDING)\
            .limit(100).stream()
        
        # Get recent speech transcriptions
        speech_transcriptions = db.collection('speech_transcriptions')\
            .where('user_id', '==', current_user_id)\
            .order_by('timestamp', direction=firestore.Query.DESCENDING)\
            .limit(100).stream()
        
        activity = []
        
        for doc in asl_detections:
            detection_data = doc.to_dict()
            detection_data['id'] = doc.id
            detection_data['type'] = 'asl_detection'
            activity.append(detection_data)
        
        for doc in speech_transcriptions:
            transcription_data = doc.to_dict()
            transcription_data['id'] = doc.id
            transcription_data['type'] = 'speech_transcription'
            activity.append(transcription_data)
        
        # Sort by timestamp
        activity.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
        
        return jsonify({
            'activity': activity[:50],  # Return top 50 activities
            'total_count': len(activity)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@user_bp.route('/delete', methods=['DELETE'])
@jwt_required()
def delete_user_account():
    """Delete user account (admin function)"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        confirmation = data.get('confirmation')
        
        if confirmation != 'DELETE_MY_ACCOUNT':
            return jsonify({'error': 'Confirmation required'}), 400
        
        db = firestore.client()
        
        # Delete user data from Firestore
        db.collection('users').document(current_user_id).delete()
        
        # Delete user's meetings, detections, etc.
        # Note: In production, you might want to anonymize data instead of deleting
        
        return jsonify({
            'message': 'Account deleted successfully'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_user_stats(user_id):
    """Helper function to get user statistics"""
    try:
        db = firestore.client()
        
        # Count ASL detections
        asl_count = len(list(db.collection('asl_detections')
            .where('user_id', '==', user_id).stream()))
        
        # Count speech transcriptions
        speech_count = len(list(db.collection('speech_transcriptions')
            .where('user_id', '==', user_id).stream()))
        
        # Count meetings participated
        meetings_count = len(list(db.collection('meetings')
            .where('participants', 'array_contains_any', [{'user_id': user_id}]).stream()))
        
        # Count meetings hosted
        hosted_count = len(list(db.collection('meetings')
            .where('host_id', '==', user_id).stream()))
        
        return {
            'asl_detections': asl_count,
            'speech_transcriptions': speech_count,
            'meetings_participated': meetings_count,
            'meetings_hosted': hosted_count,
            'total_activity': asl_count + speech_count
        }
        
    except Exception:
        return {
            'asl_detections': 0,
            'speech_transcriptions': 0,
            'meetings_participated': 0,
            'meetings_hosted': 0,
            'total_activity': 0
        } 