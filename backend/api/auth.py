from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
import firebase_admin
from firebase_admin import auth, firestore
from datetime import datetime, timedelta
import uuid

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        name = data.get('name')
        user_type = data.get('user_type', 'standard')  # standard, hearing_impaired, speech_impaired
        
        if not email or not password or not name:
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Create user in Firebase Auth
        user_record = auth.create_user(
            email=email,
            password=password,
            display_name=name
        )
        
        # Store additional user data in Firestore
        db = firestore.client()
        user_data = {
            'uid': user_record.uid,
            'email': email,
            'name': name,
            'user_type': user_type,
            'created_at': datetime.utcnow(),
            'last_login': datetime.utcnow(),
            'preferences': {
                'asl_enabled': True,
                'speech_enabled': True,
                'subtitle_enabled': True
            }
        }
        
        db.collection('users').document(user_record.uid).set(user_data)
        
        # Create JWT token
        access_token = create_access_token(
            identity=user_record.uid,
            expires_delta=timedelta(days=7)
        )
        
        return jsonify({
            'message': 'User registered successfully',
            'access_token': access_token,
            'user': {
                'uid': user_record.uid,
                'email': email,
                'name': name,
                'user_type': user_type
            }
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    """Login user"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Missing email or password'}), 400
        
        # Verify user in Firebase Auth
        user_record = auth.get_user_by_email(email)
        
        # Update last login
        db = firestore.client()
        db.collection('users').document(user_record.uid).update({
            'last_login': datetime.utcnow()
        })
        
        # Create JWT token
        access_token = create_access_token(
            identity=user_record.uid,
            expires_delta=timedelta(days=7)
        )
        
        # Get user data from Firestore
        user_doc = db.collection('users').document(user_record.uid).get()
        user_data = user_doc.to_dict()
        
        return jsonify({
            'message': 'Login successful',
            'access_token': access_token,
            'user': {
                'uid': user_record.uid,
                'email': user_data.get('email'),
                'name': user_data.get('name'),
                'user_type': user_data.get('user_type'),
                'preferences': user_data.get('preferences', {})
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': 'Invalid credentials'}), 401

@auth_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """Get current user profile"""
    try:
        current_user_id = get_jwt_identity()
        db = firestore.client()
        user_doc = db.collection('users').document(current_user_id).get()
        
        if not user_doc.exists:
            return jsonify({'error': 'User not found'}), 404
        
        user_data = user_doc.to_dict()
        return jsonify({
            'user': {
                'uid': user_data.get('uid'),
                'email': user_data.get('email'),
                'name': user_data.get('name'),
                'user_type': user_data.get('user_type'),
                'preferences': user_data.get('preferences', {}),
                'created_at': user_data.get('created_at').isoformat() if user_data.get('created_at') else None
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@auth_bp.route('/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    """Update user profile"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        db = firestore.client()
        update_data = {}
        
        if 'name' in data:
            update_data['name'] = data['name']
        if 'user_type' in data:
            update_data['user_type'] = data['user_type']
        if 'preferences' in data:
            update_data['preferences'] = data['preferences']
        
        if update_data:
            db.collection('users').document(current_user_id).update(update_data)
        
        return jsonify({'message': 'Profile updated successfully'}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    """Logout user (client should discard token)"""
    # In a more complex setup, you might want to blacklist the token
    return jsonify({'message': 'Logged out successfully'}), 200 