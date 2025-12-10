import os
import hashlib
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize Supabase client
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_ANON_KEY')

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(username, password, name, gender, grade, age):
    """Create new user account in Supabase"""
    try:
        hashed_pwd = hash_password(password)
        
        data = {
            'username': username,
            'password': hashed_pwd,
            'name': name,
            'gender': gender,
            'grade': grade,
            'age': age
        }
        
        response = supabase.table('users').insert(data).execute()
        
        return True, "Account created successfully!"
    
    except Exception as e:
        error_message = str(e)
        if 'duplicate key' in error_message.lower() or 'unique' in error_message.lower():
            return False, "Username already exists!"
        return False, f"Error: {error_message}"


def verify_user(username, password):
    """Verify user credentials from Supabase"""
    try:
        hashed_pwd = hash_password(password)
        
        response = supabase.table('users').select('*').eq('username', username).eq('password', hashed_pwd).execute()
        
        if response.data and len(response.data) > 0:
            user = response.data[0]
            return True, {
                'id': user['id'],
                'username': user['username'],
                'name': user['name'],
                'gender': user['gender'],
                'grade': user['grade'],
                'age': user['age']
            }
        
        return False, None
    
    except Exception as e:
        print(f"Verification error: {str(e)}")
        return False, None


def get_user_tests(username):
    """Get all test results for a user from Supabase"""
    try:
        response = supabase.table('test_results').select('*').eq('username', username).order('timestamp', desc=True).execute()
        
        return response.data if response.data else []
    
    except Exception as e:
        print(f"Error fetching tests: {str(e)}")
        return []


def save_test_result(username, test_data):
    """Save test result to Supabase"""
    try:
        # Get user_id
        user_response = supabase.table('users').select('id').eq('username', username).execute()
        
        if not user_response.data:
            print(f"User not found: {username}")
            return False
        
        user_id = user_response.data[0]['id']
        
        # Prepare data for insertion
        data = {
            'user_id': user_id,
            'username': username,
            'total_score': test_data.get('total_score', 0),
            'topic_scores': test_data.get('topic_scores', {}),
            'individual_analyses': test_data.get('individual_analyses', []),
            'final_feedback': test_data.get('final_feedback', ''),
            'aggregated_input': test_data.get('aggregated_input', '')
        }
        
        response = supabase.table('test_results').insert(data).execute()
        
        return True
    
    except Exception as e:
        print(f"Error saving test result: {str(e)}")
        return False


def test_connection():
    """Test Supabase connection"""
    try:
        response = supabase.table('users').select('count').execute()
        print("✅ Supabase connection successful!")
        return True
    except Exception as e:
        print(f"❌ Supabase connection failed: {str(e)}")
        return False