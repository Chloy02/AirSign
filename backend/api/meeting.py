from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime, timedelta
import uuid
import firebase_admin
from firebase_admin import firestore

meeting_bp = Blueprint('meeting', __name__)

@meeting_bp.route('/create', methods=['POST'])
@jwt_required()
def create_meeting():
    """Create a new meeting room"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        title = data.get('title', 'Untitled Meeting')
        description = data.get('description', '')
        max_participants = data.get('max_participants', 2)
        is_private = data.get('is_private', False)
        settings = data.get('settings', {
            'asl_enabled': True,
            'speech_enabled': True,
            'subtitle_enabled': True,
            'recording_enabled': False
        })
        
        # Generate unique room ID
        room_id = str(uuid.uuid4())
        
        # Get user data
        db = firestore.client()
        user_doc = db.collection('users').document(current_user_id).get()
        user_data = user_doc.to_dict()
        
        # Create meeting document
        meeting_data = {
            'room_id': room_id,
            'title': title,
            'description': description,
            'host_id': current_user_id,
            'host_name': user_data.get('name', 'Unknown'),
            'max_participants': max_participants,
            'is_private': is_private,
            'settings': settings,
            'status': 'active',
            'created_at': datetime.utcnow(),
            'started_at': datetime.utcnow(),
            'participants': [{
                'user_id': current_user_id,
                'name': user_data.get('name', 'Unknown'),
                'user_type': user_data.get('user_type', 'standard'),
                'joined_at': datetime.utcnow(),
                'is_host': True,
                'is_active': True
            }],
            'current_participant_count': 1
        }
        
        db.collection('meetings').document(room_id).set(meeting_data)
        
        return jsonify({
            'success': True,
            'room_id': room_id,
            'meeting': meeting_data,
            'message': 'Meeting created successfully'
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@meeting_bp.route('/join/<room_id>', methods=['POST'])
@jwt_required()
def join_meeting(room_id):
    """Join an existing meeting room"""
    try:
        current_user_id = get_jwt_identity()
        
        db = firestore.client()
        meeting_ref = db.collection('meetings').document(room_id)
        meeting_doc = meeting_ref.get()
        
        if not meeting_doc.exists:
            return jsonify({'error': 'Meeting not found'}), 404
        
        meeting_data = meeting_doc.to_dict()
        
        # Check if meeting is active
        if meeting_data.get('status') != 'active':
            return jsonify({'error': 'Meeting is not active'}), 400
        
        # Check participant limit
        current_count = meeting_data.get('current_participant_count', 0)
        max_participants = meeting_data.get('max_participants', 2)
        
        if current_count >= max_participants:
            return jsonify({'error': 'Meeting is full'}), 400
        
        # Check if user is already in the meeting
        participants = meeting_data.get('participants', [])
        existing_participant = next(
            (p for p in participants if p['user_id'] == current_user_id), None
        )
        
        if existing_participant:
            # Update existing participant status
            existing_participant['is_active'] = True
            existing_participant['joined_at'] = datetime.utcnow()
        else:
            # Add new participant
            user_doc = db.collection('users').document(current_user_id).get()
            user_data = user_doc.to_dict()
            
            new_participant = {
                'user_id': current_user_id,
                'name': user_data.get('name', 'Unknown'),
                'user_type': user_data.get('user_type', 'standard'),
                'joined_at': datetime.utcnow(),
                'is_host': False,
                'is_active': True
            }
            participants.append(new_participant)
            current_count += 1
        
        # Update meeting document
        meeting_ref.update({
            'participants': participants,
            'current_participant_count': current_count
        })
        
        return jsonify({
            'success': True,
            'room_id': room_id,
            'meeting': meeting_data,
            'participant': existing_participant or new_participant,
            'message': 'Joined meeting successfully'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@meeting_bp.route('/leave/<room_id>', methods=['POST'])
@jwt_required()
def leave_meeting(room_id):
    """Leave a meeting room"""
    try:
        current_user_id = get_jwt_identity()
        
        db = firestore.client()
        meeting_ref = db.collection('meetings').document(room_id)
        meeting_doc = meeting_ref.get()
        
        if not meeting_doc.exists:
            return jsonify({'error': 'Meeting not found'}), 404
        
        meeting_data = meeting_doc.to_dict()
        participants = meeting_data.get('participants', [])
        
        # Find and update participant
        for participant in participants:
            if participant['user_id'] == current_user_id:
                participant['is_active'] = False
                participant['left_at'] = datetime.utcnow()
                break
        
        current_count = sum(1 for p in participants if p.get('is_active', False))
        
        # Update meeting document
        meeting_ref.update({
            'participants': participants,
            'current_participant_count': current_count
        })
        
        # If no active participants, end the meeting
        if current_count == 0:
            meeting_ref.update({
                'status': 'ended',
                'ended_at': datetime.utcnow()
            })
        
        return jsonify({
            'success': True,
            'message': 'Left meeting successfully'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@meeting_bp.route('/<room_id>', methods=['GET'])
@jwt_required()
def get_meeting_info(room_id):
    """Get meeting information"""
    try:
        current_user_id = get_jwt_identity()
        
        db = firestore.client()
        meeting_doc = db.collection('meetings').document(room_id).get()
        
        if not meeting_doc.exists:
            return jsonify({'error': 'Meeting not found'}), 404
        
        meeting_data = meeting_doc.to_dict()
        
        # Check if user is a participant
        participants = meeting_data.get('participants', [])
        is_participant = any(p['user_id'] == current_user_id for p in participants)
        
        if not is_participant and meeting_data.get('is_private', False):
            return jsonify({'error': 'Access denied'}), 403
        
        return jsonify({
            'meeting': meeting_data
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@meeting_bp.route('/<room_id>/participants', methods=['GET'])
@jwt_required()
def get_meeting_participants(room_id):
    """Get list of meeting participants"""
    try:
        current_user_id = get_jwt_identity()
        
        db = firestore.client()
        meeting_doc = db.collection('meetings').document(room_id).get()
        
        if not meeting_doc.exists:
            return jsonify({'error': 'Meeting not found'}), 404
        
        meeting_data = meeting_doc.to_dict()
        participants = meeting_data.get('participants', [])
        
        # Filter active participants
        active_participants = [p for p in participants if p.get('is_active', False)]
        
        return jsonify({
            'participants': active_participants,
            'total_count': len(active_participants)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@meeting_bp.route('/<room_id>/webrtc-signal', methods=['POST'])
@jwt_required()
def handle_webrtc_signal(room_id):
    """Handle WebRTC signaling messages"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        signal_type = data.get('type')  # offer, answer, ice-candidate
        target_user_id = data.get('target_user_id')
        signal_data = data.get('signal_data')
        
        if not all([signal_type, target_user_id, signal_data]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Store signaling data in Firestore for real-time delivery
        db = firestore.client()
        signal_doc = {
            'room_id': room_id,
            'from_user_id': current_user_id,
            'target_user_id': target_user_id,
            'signal_type': signal_type,
            'signal_data': signal_data,
            'timestamp': datetime.utcnow(),
            'processed': False
        }
        
        db.collection('webrtc_signals').add(signal_doc)
        
        return jsonify({
            'success': True,
            'message': 'Signal sent successfully'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@meeting_bp.route('/<room_id>/settings', methods=['PUT'])
@jwt_required()
def update_meeting_settings(room_id):
    """Update meeting settings (host only)"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        db = firestore.client()
        meeting_ref = db.collection('meetings').document(room_id)
        meeting_doc = meeting_ref.get()
        
        if not meeting_doc.exists:
            return jsonify({'error': 'Meeting not found'}), 404
        
        meeting_data = meeting_doc.to_dict()
        
        # Check if user is the host
        if meeting_data.get('host_id') != current_user_id:
            return jsonify({'error': 'Only host can update settings'}), 403
        
        # Update settings
        new_settings = data.get('settings', {})
        current_settings = meeting_data.get('settings', {})
        current_settings.update(new_settings)
        
        meeting_ref.update({
            'settings': current_settings
        })
        
        return jsonify({
            'success': True,
            'settings': current_settings,
            'message': 'Settings updated successfully'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@meeting_bp.route('/user/meetings', methods=['GET'])
@jwt_required()
def get_user_meetings():
    """Get user's meetings (hosted and participated)"""
    try:
        current_user_id = get_jwt_identity()
        limit = int(request.args.get('limit', 20))
        
        db = firestore.client()
        
        # Get hosted meetings
        hosted_meetings = db.collection('meetings').where('host_id', '==', current_user_id)\
            .order_by('created_at', direction=firestore.Query.DESCENDING).limit(limit).stream()
        
        # Get participated meetings
        participated_meetings = db.collection('meetings')\
            .where('participants', 'array_contains_any', [{'user_id': current_user_id}])\
            .order_by('created_at', direction=firestore.Query.DESCENDING).limit(limit).stream()
        
        meetings = []
        
        for doc in hosted_meetings:
            meeting_data = doc.to_dict()
            meeting_data['id'] = doc.id
            meeting_data['role'] = 'host'
            meetings.append(meeting_data)
        
        for doc in participated_meetings:
            meeting_data = doc.to_dict()
            meeting_data['id'] = doc.id
            meeting_data['role'] = 'participant'
            meetings.append(meeting_data)
        
        # Sort by creation date
        meetings.sort(key=lambda x: x.get('created_at', datetime.min), reverse=True)
        
        return jsonify({
            'meetings': meetings[:limit],
            'total_count': len(meetings)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500 