import streamlit as st
import sqlite3
import hashlib
import json
import requests
import base64
from datetime import datetime
from PIL import Image
import io
import os
from dotenv import load_dotenv
import plotly.graph_objects as go
import time
from prompts import QWEN_PROMPT, DEEPSEEK_PROMPT_1, DEEPSEEK_PROMPT_2
from questions import MATH_QUESTIONS

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CONSTANTS ====================

DB_PATH = "data/users.db"
RESULTS_PATH = "data/test_results.json"
UPLOADS_FOLDER = "uploads"

# Math topics - 2 questions each = 10 total
MATH_TOPICS = [
    "Algebra",
    "Rational Number System",
    "Ratio and Proportion",
    "Percentage",
    "Geometry"
]

TOPIC_DESCRIPTIONS = {
    "Algebra": "Linear Equations, Quadratic Equations, Polynomials",
    "Rational Number System": "Rational Numbers, Operations, Properties",
    "Ratio and Proportion": "Ratios, Proportions, Direct/Inverse Variation",
    "Percentage": "Percentage Calculations, Applications, Problems",
    "Geometry": "Shapes, Area, Perimeter, Volume, Theorems"
}

# API Configuration
QWEN_API_ENDPOINT = os.getenv('QWEN_API_ENDPOINT')
QWEN_API_KEY = os.getenv('QWEN_API_KEY')
QWEN_MODEL = os.getenv('QWEN_MODEL', 'qwen/qwen-2-vl-72b-instruct')

DEEPSEEK_API_ENDPOINT = os.getenv('DEEPSEEK_API_ENDPOINT')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs(UPLOADS_FOLDER, exist_ok=True)

# ==================== DATABASE FUNCTIONS ====================

def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT NOT NULL,
            gender TEXT NOT NULL,
            grade TEXT NOT NULL,
            age INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()


def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(username, password, name, gender, grade, age):
    """Create new user account"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        hashed_pwd = hash_password(password)
        cursor.execute('''
            INSERT INTO users (username, password, name, gender, grade, age)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (username, hashed_pwd, name, gender, grade, age))

        conn.commit()
        conn.close()
        return True, "Account created successfully!"
    except sqlite3.IntegrityError:
        return False, "Username already exists!"
    except Exception as e:
        return False, f"Error: {str(e)}"


def verify_user(username, password):
    """Verify user credentials"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        hashed_pwd = hash_password(password)
        cursor.execute('''
            SELECT id, name, gender, grade, age FROM users 
            WHERE username = ? AND password = ?
        ''', (username, hashed_pwd))

        result = cursor.fetchone()
        conn.close()

        if result:
            return True, {
                'id': result[0],
                'username': username,
                'name': result[1],
                'gender': result[2],
                'grade': result[3],
                'age': result[4]
            }
        return False, None
    except Exception:
        return False, None

# ==================== TEST RESULTS FUNCTIONS ====================

def get_user_tests(username):
    """Get all test results for a user"""
    try:
        if not os.path.exists(RESULTS_PATH):
            return []

        with open(RESULTS_PATH, 'r') as f:
            all_results = json.load(f)

        user_results = [r for r in all_results if r.get('username') == username]
        return sorted(user_results, key=lambda x: x.get('timestamp', ''), reverse=True)
    except:
        return []


def save_test_result(username, test_data):
    """Save test result to JSON file"""
    try:
        if os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH, 'r') as f:
                results = json.load(f)
        else:
            results = []

        test_data['username'] = username
        test_data['timestamp'] = datetime.now().isoformat()
        results.append(test_data)

        with open(RESULTS_PATH, 'w') as f:
            json.dump(results, f, indent=2)

        return True
    except Exception as e:
        st.error(f"Error saving result: {str(e)}")
        return False

# ==================== AI PROCESSING FUNCTIONS ====================

def process_image_with_qwen(image_base64, prompt, stream_placeholder=None):
    """
    Process image with Qwen-VL API and stream only the model output.
    """
    try:
        headers = {"Authorization": f"Bearer {QWEN_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": QWEN_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }
            ],
            "stream": False
        }

        resp = requests.post(QWEN_API_ENDPOINT, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        output = resp.json()['choices'][0]['message']['content']

        # Simulated streaming of the final text
        if stream_placeholder:
            buffer = ""
            for ch in output:
                buffer += ch
                stream_placeholder.markdown(buffer)
                time.sleep(0.01)

        return output

    except Exception as e:
        msg = f"[Qwen Error: {str(e)}]"
        if stream_placeholder:
            stream_placeholder.error(msg)
        return msg


def process_with_deepseek(input_text, system_prompt, stream_placeholder=None, model_name="DeepSeek", model_id=None):
    """
    Process text with DeepSeek API and stream only the model output.
    """
    try:
        selected_model = model_id or DEEPSEEK_MODEL
        headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}
            ],
            "temperature": 0.7,
            "stream": False
        }

        resp = requests.post(DEEPSEEK_API_ENDPOINT, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        output = resp.json()['choices'][0]['message']['content']

        # Simulated streaming of the final text
        if stream_placeholder:
            buffer = ""
            for ch in output:
                buffer += ch
                stream_placeholder.markdown(buffer)
                time.sleep(0.01)

        return output

    except Exception as e:
        msg = f"[DeepSeek Error: {str(e)}]"
        if stream_placeholder:
            stream_placeholder.error(msg)
        return msg

# ==================== ANALYSIS PIPELINE (STACKED OUTPUTS PER QUESTION) ====================

def analyze_test_images(questions_data):
    """
    Runs the full pipeline and renders EVERYTHING on one page:
    For each question:
      - Qwen output box (streams)
      - DeepSeek output box (streams)
    """
    from questions import MATH_QUESTIONS

    all_analyses = []
    total_score = 0
    topic_scores = {topic: 0 for topic in MATH_TOPICS}

    st.title("🧪 AI Analysis — All Questions")

    progress_bar = st.progress(0)
    status_text = st.empty()

    total_qs = len(questions_data)

    for idx, q in enumerate(questions_data, start=1):
        topic = q['topic']
        qnum = q['question_num']
        imgs = q['images']
        pages = len(imgs)

        # --- Question header ---
        st.markdown(f"## Q{idx}: {topic} — Question {qnum}")

        # --- Qwen box (streams) ---
        st.markdown("**Qwen (extraction / understanding):**")
        qwen_box = st.empty()
        extracted_pages = []

        for p_i, img_bytes in enumerate(imgs, start=1):
            status_text.text(f"Q{idx}: extracting page {p_i}/{pages} with Qwen…")
            b64 = base64.b64encode(img_bytes).decode('utf-8')
            page_txt = process_image_with_qwen(b64, QWEN_PROMPT, qwen_box)
            extracted_pages.append(f"--- Page {p_i} ---\n{page_txt}")

        combined_extraction = "\n\n".join(extracted_pages)
        if pages > 1:
            combined_extraction = f"[MULTI-PAGE SOLUTION - {pages} pages]\n\n{combined_extraction}"

        # --- DeepSeek box (streams) ---
        st.markdown("**DeepSeek (analysis & scoring):**")
        deepseek_box = st.empty()
        status_text.text(f"Q{idx}: analyzing with DeepSeek…")

        actual_q = MATH_QUESTIONS[topic][f"question_{qnum}"]
        deepseek_input = f"""ORIGINAL QUESTION:
{actual_q}

STUDENT'S EXTRACTED SOLUTION (Complete):
{combined_extraction}"""

        deepseek_out = process_with_deepseek(
            deepseek_input,
            DEEPSEEK_PROMPT_1,
            deepseek_box,   # streams here
            "DeepSeek #1 (Scorer)"
        )

        # --- scoring (same logic as before) ---
        score = 1 if ("SCORE: 1" in deepseek_out or "CORRECT: YES" in deepseek_out.upper()) else 0
        total_score += score
        topic_scores[topic] += score

        all_analyses.append({
            'topic': topic,
            'question_num': qnum,
            'num_pages': pages,
            'qwen_output': combined_extraction,
            'deepseek_output': deepseek_out,
            'score': score
        })

        st.markdown("---")
        progress_bar.progress(idx / total_qs)

    # --- Final feedback (optional summary using DeepSeek #2) ---
    status_text.text("Generating comprehensive feedback…")
    aggregated_input = "=== STUDENT TEST ANALYSIS - ALL RESPONSES ===\n\n"
    for a in all_analyses:
        header = f"--- {a['topic']} - Question {a['question_num']}"
        if a['num_pages'] > 1:
            header += f" ({a['num_pages']} pages)"
        header += " ---\n"
        aggregated_input += header
        aggregated_input += f"Score: {a['score']}/1\n"
        aggregated_input += a['deepseek_output'] + "\n\n" + "="*50 + "\n\n"

    final_feedback = process_with_deepseek(
        aggregated_input, DEEPSEEK_PROMPT_2, None, "DeepSeek #2 (Final Analyzer)"
    )

    progress_bar.empty()
    status_text.empty()

    return {
        'total_score': total_score,
        'topic_scores': topic_scores,
        'individual_analyses': all_analyses,
        'final_feedback': final_feedback,
        'aggregated_input': aggregated_input
    }

# ==================== SESSION STATE INITIALIZATION ====================

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'page': 'login',
        'logged_in': False,
        'user': None,
        'test_images': {}
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ==================== PAGE FUNCTIONS ====================

def login_page():
    """Login page UI"""
    st.title("🎓 AI Math Tutor")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### Login to Your Account")
        st.markdown("---")

        with st.form("login_form"):
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")

            col_a, col_b = st.columns(2)

            with col_a:
                submitted = st.form_submit_button("Login", use_container_width=True)

            with col_b:
                signup_btn = st.form_submit_button("Sign Up", use_container_width=True)

            if submitted:
                if username and password:
                    success, user_data = verify_user(username, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user = user_data
                        st.session_state.page = 'dashboard'
                        st.success("✅ Login successful!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password!")
                else:
                    st.warning("⚠️ Please fill in all fields!")

            if signup_btn:
                st.session_state.page = 'signup'
                st.rerun()


def signup_page():
    """Sign up page UI"""
    st.title("🎓 AI Tutor")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### 📝 Student Registration")
        st.markdown("---")

        with st.form("signup_form"):
            st.markdown("**🔐 Account Credentials**")
            username = st.text_input("Username", placeholder="Choose a unique username")
            password = st.text_input("Password", type="password", placeholder="Min 6 characters")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")

            st.markdown("---")
            st.markdown("**👤 Student Information**")

            name = st.text_input("Full Name", placeholder="Enter your full name")

            col_x, col_y = st.columns(2)
            with col_x:
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            with col_y:
                age = st.number_input("Age", min_value=13, max_value=25, value=16)

            grade = st.selectbox("Grade Level", ["O Level", "A Level"])

            st.markdown("---")

            col_a, col_b = st.columns(2)

            with col_a:
                submitted = st.form_submit_button("Create Account", use_container_width=True)

            with col_b:
                back_btn = st.form_submit_button("← Back to Login", use_container_width=True)

            if submitted:
                if not all([username, password, confirm_password, name]):
                    st.error("❌ Please fill in all fields!")
                elif len(username) < 3:
                    st.error("❌ Username must be at least 3 characters!")
                elif len(password) < 6:
                    st.error("❌ Password must be at least 6 characters!")
                elif password != confirm_password:
                    st.error("❌ Passwords do not match!")
                else:
                    success, message = create_user(username, password, name, gender, grade, age)
                    if success:
                        st.success(f"✅ {message}")
                        st.info("Please login with your credentials.")
                        time.sleep(2)
                        st.session_state.page = 'login'
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")

            if back_btn:
                st.session_state.page = 'login'
                st.rerun()


def parse_ai_feedback(feedback_text):
    """Parse structured AI feedback into sections"""
    sections = {
        'overall': '',
        'strong_topics': [],
        'weak_topics': [],
        'recommendations': '',
        'encouragement': ''
    }

    try:
        if 'OVERALL SUMMARY:' in feedback_text:
            overall = feedback_text.split('OVERALL SUMMARY:')[1].split('STRONG TOPICS:')[0].strip()
            sections['overall'] = overall

        if 'STRONG TOPICS:' in feedback_text:
            strong = feedback_text.split('STRONG TOPICS:')[1].split('WEAK TOPICS:')[0].strip()
            sections['strong_topics'] = [line.strip('- ').strip() for line in strong.split('\n') if line.strip().startswith('-')]

        if 'WEAK TOPICS:' in feedback_text:
            weak = feedback_text.split('WEAK TOPICS:')[1].split('RECOMMENDATIONS:')[0].strip()
            sections['weak_topics'] = [line.strip('- ').strip() for line in weak.split('\n') if line.strip().startswith('-')]

        if 'RECOMMENDATIONS:' in feedback_text:
            recommendations = feedback_text.split('RECOMMENDATIONS:')[1].split('ENCOURAGEMENT:')[0].strip()
            sections['recommendations'] = recommendations

        if 'ENCOURAGEMENT:' in feedback_text:
            encouragement = feedback_text.split('ENCOURAGEMENT:')[1].strip()
            sections['encouragement'] = encouragement

    except Exception:
        sections['overall'] = feedback_text

    return sections


def dashboard_page():
    """Main dashboard page with statistics and graphs"""
    user = st.session_state.user

    # Sidebar navigation
    with st.sidebar:
        st.markdown(f"### 👤 Hello, {user['name']}!")
        st.markdown(f"**Grade:** {user['grade']}")
        st.markdown(f"**Age:** {user['age']}")
        st.markdown("---")

        if st.button("📝 Take New Test", use_container_width=True, type="primary"):
            st.session_state.page = 'test'
            st.session_state.test_images = {}  # Reset test images
            st.rerun()

        st.markdown("---")

        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Main content
    st.title("📊 Your Progress Dashboard")

    # Get user's test history
    tests = get_user_tests(user['username'])

    if not tests:
        st.info("👋 Welcome! You haven't taken any tests yet.")
        st.markdown("### 🚀 Get Started")
        st.markdown("Click **'Take New Test'** in the sidebar to begin your first mathematics assessment!")

        st.markdown("---")
        st.markdown("### 📚 Topics Covered in Each Test")
        st.markdown("*Each test contains 10 questions (2 per topic)*")

        cols = st.columns(3)
        for idx, topic in enumerate(MATH_TOPICS):
            with cols[idx % 3]:
                st.markdown(f"**{topic}**")
                st.caption(TOPIC_DESCRIPTIONS[topic])

    else:
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("📊 Total Tests", len(tests))

        with col2:
            avg_score = sum(t.get('total_score', 0) for t in tests) / len(tests)
            st.metric("📈 Average Score", f"{avg_score:.1f}/10")

        with col3:
            latest_score = tests[0].get('total_score', 0)
            st.metric("🎯 Latest Score", f"{latest_score}/10")

        with col4:
            avg_percentage = (avg_score / 10) * 100
            st.metric("✨ Average %", f"{avg_percentage:.0f}%")

        st.markdown("---")

        # Progress over time
        if len(tests) > 1:
            st.markdown("### 📈 Score Progress Over Time")

            dates = [datetime.fromisoformat(t['timestamp']).strftime('%b %d') for t in reversed(tests)]
            scores = [t.get('total_score', 0) for t in reversed(tests)]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=scores,
                mode='lines+markers',
                name='Score',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=12, color='#1f77b4'),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.1)'
            ))

            fig.update_layout(
                xaxis_title="Test Date",
                yaxis_title="Score (out of 10)",
                yaxis=dict(range=[0, 11]),
                height=400,
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

        # Topic-wise performance (numeric bar chart)
        st.markdown("### 📊 Topic-wise Performance")

        # Aggregate topic scores
        topic_scores = {topic: [] for topic in MATH_TOPICS}
        for test in tests:
            topics = test.get('topic_scores', {})
            for topic, score in topics.items():
                if topic in topic_scores:
                    topic_scores[topic].append(score)

        # Calculate averages
        topic_averages = {
            topic: (sum(scores) / len(scores)) if scores else 0
            for topic, scores in topic_scores.items()
        }

        # Bar chart
        colors = ['#ff7f0e' if v < 1 else '#ffd700' if v < 1.5 else '#2ca02c' for v in topic_averages.values()]

        fig = go.Figure(data=[
            go.Bar(
                x=list(topic_averages.keys()),
                y=list(topic_averages.values()),
                marker_color=colors,
                text=[f"{v:.1f}/2" for v in topic_averages.values()],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Average: %{y:.2f}/2<extra></extra>'
            )
        ])

        fig.update_layout(
            xaxis_title="Topic",
            yaxis_title="Average Score (out of 2)",
            yaxis=dict(range=[0, 2.5]),
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # AI-Powered Performance Analysis
        st.markdown("---")
        st.markdown("### 🤖 AI Tutor Analysis")

        # Get latest test feedback
        latest_test = tests[0]
        if latest_test.get('final_feedback'):
            # Parse AI feedback
            ai_analysis = parse_ai_feedback(latest_test['final_feedback'])

            # Overall Summary
            if ai_analysis['overall']:
                st.markdown("#### 📋 Overall Performance")
                st.info(ai_analysis['overall'])

            # Strong and Weak Topics Side by Side
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### ✅ Strong Areas")
                if ai_analysis['strong_topics']:
                    for topic in ai_analysis['strong_topics']:
                        st.success(f"✓ {topic}")
                else:
                    st.info("Keep practicing to build your strengths!")

            with col2:
                st.markdown("#### ⚠️ Areas to Improve")
                if ai_analysis['weak_topics']:
                    for topic in ai_analysis['weak_topics']:
                        st.warning(f"⚠ {topic}")
                else:
                    st.success("💪 All topics are strong!")

            # Recommendations
            if ai_analysis['recommendations']:
                st.markdown("#### 📚 Study Recommendations")
                st.markdown(ai_analysis['recommendations'])

            # Encouragement
            if ai_analysis['encouragement']:
                st.markdown("#### 💬 Message from Your AI Tutor")
                st.success(ai_analysis['encouragement'])
        else:
            st.warning("⚠️ AI analysis not available for latest test. Take a new test to see AI feedback!")

        # Recent test results
        st.markdown("---")
        st.markdown("### 📝 Recent Test Results")

        for idx, test in enumerate(tests[:3], 1):
            test_date = datetime.fromisoformat(test['timestamp']).strftime('%B %d, %Y at %H:%M')
            score = test.get('total_score', 0)
            percentage = (score / 10) * 100

            with st.expander(f"Test #{idx} - {test_date} - Score: {score}/10 ({percentage:.0f}%)"):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown("**Topic Scores:**")
                    for topic in MATH_TOPICS:
                        topic_score = test.get('topic_scores', {}).get(topic, 0)
                        emoji = "✅" if topic_score == 2 else "⚠️" if topic_score == 1 else "❌"
                        st.write(f"{emoji} {topic}: {topic_score}/2")

                with col2:
                    st.metric("Total Score", f"{score}/10")
                    st.metric("Percentage", f"{percentage:.0f}%")

                if test.get('final_feedback'):
                    st.markdown("---")
                    st.markdown("**🤖 Full AI Feedback:**")
                    st.info(test['final_feedback'])


def test_page():
    """Upload UI + single-page analysis (Q1..Q10 stacked)."""
    user = st.session_state.user

    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👤 {user['name']}")
        st.markdown(f"**📚 Grade:** {user['grade']}")
        st.markdown("---")
        if st.button("← Back to Dashboard"):
            st.session_state.page = 'dashboard'
            st.rerun()

    # Main content
    st.title("📝 Mathematics Test - O/A Level")

    st.info("""
    **📋 Test Instructions:**
    - This test contains **10 questions** (2 from each topic)
    - Upload **one or more images** for each question (for multi-page solutions)
    - When you click **Analyze**, you'll see outputs for **all questions on this page**:
      first the **Qwen** extraction, then the **DeepSeek** analysis for each question.
    """)

    st.markdown("---")

    # Initialize test_images in session state if not exists
    if 'test_images' not in st.session_state:
        st.session_state.test_images = {}

    # Create upload sections for each topic
    for topic in MATH_TOPICS:
        st.markdown(f"### 📐 {topic}")
        st.caption(TOPIC_DESCRIPTIONS[topic])

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Question 1:**")
            st.info(MATH_QUESTIONS[topic]["question_1"])

            key1 = f"{topic}_q1"
            uploaded_files1 = st.file_uploader(
                f"Upload solution (multiple images for multi-page)",
                type=['png', 'jpg', 'jpeg'],
                key=key1,
                accept_multiple_files=True
            )
            if uploaded_files1:
                st.session_state.test_images[key1] = {
                    'topic': topic,
                    'question_num': 1,
                    'images': [file.read() for file in uploaded_files1]
                }
                st.success(f"✅ Uploaded {len(uploaded_files1)} image(s)")
                
                # Show previews
                for img_idx, img_bytes in enumerate(st.session_state.test_images[key1]['images'], 1):
                    st.caption(f"Page {img_idx}")
                    image = Image.open(io.BytesIO(img_bytes))
                    st.image(image, use_column_width=True)

        with col2:
            st.markdown(f"**Question 2:**")
            st.info(MATH_QUESTIONS[topic]["question_2"])

            key2 = f"{topic}_q2"
            uploaded_files2 = st.file_uploader(
                f"Upload solution (multiple images for multi-page)",
                type=['png', 'jpg', 'jpeg'],
                key=key2,
                accept_multiple_files=True
            )
            if uploaded_files2:
                st.session_state.test_images[key2] = {
                    'topic': topic,
                    'question_num': 2,
                    'images': [file.read() for file in uploaded_files2]
                }
                st.success(f"✅ Uploaded {len(uploaded_files2)} image(s)")
                
                # Show previews
                for img_idx, img_bytes in enumerate(st.session_state.test_images[key2]['images'], 1):
                    st.caption(f"Page {img_idx}")
                    image = Image.open(io.BytesIO(img_bytes))
                    st.image(image, use_column_width=True)

        st.markdown("---")

    # Submit button
    st.markdown("### 🎯 Ready to Analyze?")

    uploaded_count = len(st.session_state.test_images)
    total_images = sum(len(q['images']) for q in st.session_state.test_images.values())
    st.write(f"**Uploaded: {uploaded_count}/10 questions ({total_images} total images)**")

    if uploaded_count < 10:
        st.warning(f"⚠️ Please upload images for all 10 questions. {10 - uploaded_count} remaining.")

    analyze = st.button("🚀 Analyze (show Q1→Q10 outputs here)", type="primary", disabled=(uploaded_count < 10))

    if analyze:
        # deterministic order: topic order then q1, q2
        ordered_questions = []
        for topic in MATH_TOPICS:
            for qn in (1, 2):
                k = f"{topic}_q{qn}"
                ordered_questions.append(st.session_state.test_images[k])

        # Run analysis and render outputs on THIS page
        result = analyze_test_images(ordered_questions)

        # Save & back actions
        st.markdown("### 💾 Save or Go Back")
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("💾 Save result to history"):
                if save_test_result(user['username'], result):
                    st.success("Saved.")
        with c2:
            if st.button("⬅️ Back to dashboard"):
                st.session_state.page = 'dashboard'
                st.rerun()

# ==================== MAIN APP ====================

def main():
    """Main application router"""

    # Initialize database
    init_db()

    # Initialize session state
    init_session_state()

    # Route to appropriate page
    if not st.session_state.logged_in:
        # Not logged in - show auth pages
        if st.session_state.page == 'signup':
            signup_page()
        else:
            login_page()
    else:
        # Logged in - show app pages
        if st.session_state.page == 'dashboard':
            dashboard_page()
        elif st.session_state.page == 'test':
            test_page()
        else:
            dashboard_page()


if __name__ == "__main__":
    main()
