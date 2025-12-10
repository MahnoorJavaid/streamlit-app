from supabase_db import test_connection, create_user, verify_user

# Test 1: Connection
print("Testing Supabase connection...")
test_connection()

# Test 2: Create a test user
print("\nTesting user creation...")
success, message = create_user(
    username="testuser123",
    password="testpass123",
    name="Test Student",
    gender="Male",
    grade="O Level",
    age=16
)
print(f"Create user: {message}")

# Test 3: Verify the user
print("\nTesting user verification...")
success, user_data = verify_user("testuser123", "testpass123")
if success:
    print(f"✅ Login successful! User: {user_data['name']}")
else:
    print("❌ Login failed!")

print("\n✅ All tests complete!")
