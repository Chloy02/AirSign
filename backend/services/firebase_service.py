import firebase_admin
from firebase_admin import credentials, firestore, auth
import os
from dotenv import load_dotenv

load_dotenv()

def initialize_firebase():
    """Initialize Firebase Admin SDK"""
    try:
        # Check if Firebase is already initialized
        if not firebase_admin._apps:
            # Use service account key if available
            service_account_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH')
            
            if service_account_path and os.path.exists(service_account_path):
                cred = credentials.Certificate(service_account_path)
                firebase_admin.initialize_app(cred)
            else:
                # Use default credentials (for development)
                firebase_admin.initialize_app()
        
        print("Firebase initialized successfully")
        return True
        
    except Exception as e:
        print(f"Error initializing Firebase: {e}")
        return False

def get_firestore_client():
    """Get Firestore client instance"""
    try:
        return firestore.client()
    except Exception as e:
        print(f"Error getting Firestore client: {e}")
        return None

def verify_firebase_token(id_token):
    """Verify Firebase ID token"""
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token
    except Exception as e:
        print(f"Error verifying Firebase token: {e}")
        return None

def create_user_in_firebase(email, password, display_name=None):
    """Create a new user in Firebase Auth"""
    try:
        user_record = auth.create_user(
            email=email,
            password=password,
            display_name=display_name
        )
        return user_record
    except Exception as e:
        print(f"Error creating Firebase user: {e}")
        return None

def get_user_by_email(email):
    """Get user by email from Firebase Auth"""
    try:
        user_record = auth.get_user_by_email(email)
        return user_record
    except Exception as e:
        print(f"Error getting Firebase user: {e}")
        return None

def update_user_profile(uid, **kwargs):
    """Update user profile in Firebase Auth"""
    try:
        auth.update_user(uid, **kwargs)
        return True
    except Exception as e:
        print(f"Error updating Firebase user: {e}")
        return False

def delete_user_from_firebase(uid):
    """Delete user from Firebase Auth"""
    try:
        auth.delete_user(uid)
        return True
    except Exception as e:
        print(f"Error deleting Firebase user: {e}")
        return False 