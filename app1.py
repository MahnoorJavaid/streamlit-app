import streamlit as st
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
from supabase_db import create_user, verify_user, get_user_tests, save_test_result

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Math Tutor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    /* Remove default Streamlit styling */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Professional color scheme */
    :root {
        --primary-color: #2C3E50;
        --secondary-color: #3498DB;
        --success-color: #27AE60;
        --warning-color: #F39C12;
        --danger-color: #E74C3C;
        --bg-light: #ECF0F1;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: var(--primary-color);
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 4px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    /* Cards */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid var(--secondary-color);
    }
    
    /* Remove emoji/icon spacing */
    .stMarkdown p {
        line-height: 1.6;
    }
    
    /* Professional tables */
    .dataframe {
        border: 1px solid #ddd;
        border-radius: 4px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* AI Log boxes */
    .ai-log-box {
        background: #f8f9fa;
        border-left: 4px solid #3498DB;
        padding: 15px;
        margin: 10px 0;
        border-radius: 4px;
    }
    
    .ai-log-title {
        font-weight: 600;
        color: #2C3E50;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== CONSTANTS ====================

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
os.makedirs(UPLOADS_FOLDER, exist_ok=True)

# ==================== AI PROCESSING FUNCTIONS ====================

# MODIFIED PROMPT FOR QWEN - YOU NEED TO UPDATE prompts.py
QWEN_DETECTION_PROMPT = """Analyze this image and extract:
1. The QUESTION NUMBER (e.g., "Question 1", "Q1", "1)", etc.)
2. The TOPIC if mentioned (Algebra, Geometry, Percentage, etc.)
3. The complete solution written by the student
4. Whether this appears to be a CONTINUATION PAGE (no question number, continuing from previous work)

Format your response as:
QUESTION_NUMBER: [detected number or "UNKNOWN"]
TOPIC: [detected topic or "UNKNOWN"]
IS_CONTINUATION: [YES/NO]
SOLUTION:
[extracted solution text]
"""


def process_image_with_qwen(image_base64, prompt):
    """Process image with Qwen-VL API"""
    try:
        headers = {
            "Authorization": f"Bearer {QWEN_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": QWEN_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }
            ],
            "stream": False
        }

        response = requests.post(
            QWEN_API_ENDPOINT, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()
        return result['choices'][0]['message']['content']

    except Exception as e:
        return f"[Qwen Error: {str(e)}]"


def process_with_deepseek(input_text, system_prompt):
    """Process text with DeepSeek API"""
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}
            ],
            "temperature": 0.7,
            "stream": False
        }

        response = requests.post(
            DEEPSEEK_API_ENDPOINT, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()
        return result['choices'][0]['message']['content']

    except Exception as e:
        return f"[DeepSeek Error: {str(e)}]"


def parse_qwen_detection(qwen_output):
    """Parse Qwen output to extract question number and solution"""
    result = {
        'question_number': 'UNKNOWN',
        'topic': 'UNKNOWN',
        'is_continuation': False,
        'solution': qwen_output
    }

    try:
        lines = qwen_output.split('\n')
        for line in lines:
            if line.startswith('QUESTION_NUMBER:'):
                result['question_number'] = line.split(':', 1)[1].strip()
            elif line.startswith('TOPIC:'):
                result['topic'] = line.split(':', 1)[1].strip()
            elif line.startswith('IS_CONTINUATION:'):
                result['is_continuation'] = 'YES' in line.upper()
            elif line.startswith('SOLUTION:'):
                # Get everything after SOLUTION:
                idx = qwen_output.find('SOLUTION:')
                result['solution'] = qwen_output[idx + 9:].strip()
                break
    except Exception as e:
        # If parsing fails, return raw output as solution
        pass

    return result


def detect_and_group_images(uploaded_images, log_container):
    """
    Detect question numbers in images and group them accordingly
    Returns: dict mapping question keys to list of images
    """
    from questions import MATH_QUESTIONS

    with log_container:
        st.markdown("## Step 1: AI Detection of Question Numbers")
        st.info(
            f"Analyzing {len(uploaded_images)} uploaded image(s) to detect question numbers...")

    detected_images = []

    # Process each image with Qwen for detection
    for img_idx, image_bytes in enumerate(uploaded_images, 1):
        with log_container:
            st.markdown(
                f"### Analyzing Image {img_idx}/{len(uploaded_images)}")
            detection_status = st.empty()

        detection_status.info(
            f"📊 Detecting question number in image {img_idx}...")

        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        qwen_output = process_image_with_qwen(
            image_base64, QWEN_DETECTION_PROMPT)
        parsed = parse_qwen_detection(qwen_output)

        detected_images.append({
            'image_idx': img_idx,
            'image_bytes': image_bytes,
            'detected_question': parsed['question_number'],
            'detected_topic': parsed['topic'],
            'is_continuation': parsed['is_continuation'],
            'solution': parsed['solution'],
            'raw_output': qwen_output
        })

        with log_container:
            if parsed['question_number'] != 'UNKNOWN':
                detection_status.success(
                    f"✅ Detected: Question {parsed['question_number']} | Topic: {parsed['topic']}")
            elif parsed['is_continuation']:
                detection_status.warning(
                    f"⚠️ Detected: Continuation page (no question number)")
            else:
                detection_status.error(f"❌ Could not detect question number")

            with st.expander(f"View raw detection output for Image {img_idx}"):
                st.code(qwen_output, language="text")

    # Group images by question number
    with log_container:
        st.markdown("---")
        st.markdown("## Step 2: Grouping Images by Question")

    grouped_questions = {}
    current_question = None

    for img_data in detected_images:
        if img_data['detected_question'] != 'UNKNOWN':
            # New question detected
            current_question = img_data['detected_question']
            if current_question not in grouped_questions:
                grouped_questions[current_question] = []
            grouped_questions[current_question].append(img_data)
        elif img_data['is_continuation'] and current_question:
            # Continuation of previous question
            grouped_questions[current_question].append(img_data)
        else:
            # Unidentified image - create special group
            if 'UNIDENTIFIED' not in grouped_questions:
                grouped_questions['UNIDENTIFIED'] = []
            grouped_questions['UNIDENTIFIED'].append(img_data)

    # Display grouping results
    with log_container:
        st.markdown("### Grouping Results:")

        for q_num, images in grouped_questions.items():
            if q_num == 'UNIDENTIFIED':
                st.error(
                    f"❌ Question: UNIDENTIFIED - {len(images)} image(s) could not be matched")
            else:
                st.success(
                    f"✅ Question {q_num}: {len(images)} image(s) grouped together")

        st.markdown("---")

    return grouped_questions, detected_images


def map_to_test_structure(grouped_questions, log_container):
    """
    Map detected questions to actual test structure (5 topics x 2 questions)
    Returns: list of dicts with {topic, question_num, images}
    """
    from questions import MATH_QUESTIONS

    with log_container:
        st.markdown("## Step 3: Mapping to Test Structure")
        st.info("Matching detected questions to the 10-question test format...")

    # Create mapping of question numbers to test structure
    # Expected: Q1-Q10 or Question 1-10
    test_questions = []
    mapped_count = 0

    for topic_idx, topic in enumerate(MATH_TOPICS):
        for q_num in [1, 2]:
            # Calculate expected question number (1-10)
            expected_q_num = (topic_idx * 2) + q_num

            # Try to find matching detected question
            found = False
            for detected_q_key in grouped_questions.keys():
                if detected_q_key == 'UNIDENTIFIED':
                    continue

                # Try to extract number from detected question
                detected_num_str = ''.join(filter(str.isdigit, detected_q_key))
                if detected_num_str and int(detected_num_str) == expected_q_num:
                    # Found match
                    images_list = [img['image_bytes']
                                   for img in grouped_questions[detected_q_key]]
                    test_questions.append({
                        'topic': topic,
                        'question_num': q_num,
                        'images': images_list,
                        'detected_key': detected_q_key
                    })
                    mapped_count += 1
                    found = True
                    break

            if not found:
                # Question not found
                test_questions.append({
                    'topic': topic,
                    'question_num': q_num,
                    'images': [],
                    'detected_key': None
                })

    # Display mapping results
    with log_container:
        st.markdown("### Mapping Results:")
        st.info(f"Successfully mapped {mapped_count}/10 questions")

        # Show detailed mapping
        for idx, q_data in enumerate(test_questions, 1):
            if q_data['images']:
                st.success(
                    f"✅ Q{idx} ({q_data['topic']} - Q{q_data['question_num']}): {len(q_data['images'])} image(s) - Detected as '{q_data['detected_key']}'")
            else:
                st.error(
                    f"❌ Q{idx} ({q_data['topic']} - Q{q_data['question_num']}): NO IMAGES FOUND")

        # Handle unidentified images
        if 'UNIDENTIFIED' in grouped_questions:
            st.warning(
                f"⚠️ WARNING: {len(grouped_questions['UNIDENTIFIED'])} image(s) could not be identified")
            with st.expander("View unidentified images"):
                for img_data in grouped_questions['UNIDENTIFIED']:
                    st.markdown(f"**Image {img_data['image_idx']}:**")
                    st.code(img_data['raw_output'], language="text")

        st.markdown("---")

    return test_questions


def analyze_test_images_with_streaming(questions_data, log_container):
    """
    Analyze all test images through AI pipeline with real-time streaming logs
    questions_data: list of dicts with {topic, question_num, images: [image_bytes, ...]}
    log_container: Streamlit container for displaying logs
    """
    from questions import MATH_QUESTIONS

    all_analyses = []
    total_score = 0
    topic_scores = {topic: 0 for topic in MATH_TOPICS}

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    with log_container:
        st.markdown("## Step 4: Analyzing Solutions & Scoring")
        st.markdown("---")

    # Process each question
    valid_questions = [q for q in questions_data if q['images']]

    for idx, question_data in enumerate(valid_questions):
        topic = question_data['topic']
        question_num = question_data['question_num']
        images_list = question_data['images']
        num_images = len(images_list)

        # Create question section in log
        with log_container:
            st.markdown(f"### Question: {topic} - Q{question_num}")
            st.markdown(f"**Pages uploaded:** {num_images}")
            st.markdown("---")

            # Step: Qwen Extraction (already done in detection phase, just consolidate)
            qwen_container = st.container()
            with qwen_container:
                st.markdown("#### 🔍 Qwen-VL Text Extraction")
                qwen_status = st.empty()
                qwen_output_box = st.empty()

        status_text.text(
            f"Processing {topic} - Question {question_num} (Extracting {num_images} page(s))...")

        # Extract text from all images
        extracted_texts = []
        for img_idx, image_bytes in enumerate(images_list, 1):
            qwen_status.info(f"📄 Extracting page {img_idx}/{num_images}...")

            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            qwen_output = process_image_with_qwen(image_base64, QWEN_PROMPT)
            extracted_texts.append(f"--- Page {img_idx} ---\n{qwen_output}")

        combined_extraction = "\n\n".join(extracted_texts)
        if num_images > 1:
            combined_extraction = f"[MULTI-PAGE SOLUTION - {num_images} pages]\n\n{combined_extraction}"

        # Display Qwen output
        with log_container:
            qwen_status.success(
                f"✅ Extraction complete ({num_images} page(s))")
            with qwen_output_box.container():
                st.code(combined_extraction, language="text")

        # DeepSeek Analysis
        with log_container:
            st.markdown("#### 🤖 DeepSeek Analysis & Scoring")
            deepseek_status = st.empty()
            deepseek_output_box = st.empty()

        status_text.text(
            f"Processing {topic} - Question {question_num} (Analyzing solution)...")
        deepseek_status.info("🔬 Analyzing student's solution...")

        # Get actual question
        actual_question = MATH_QUESTIONS[topic][f"question_{question_num}"]
        deepseek_input = f"""ORIGINAL QUESTION:
{actual_question}

STUDENT'S EXTRACTED SOLUTION (Complete):
{combined_extraction}"""

        deepseek_output = process_with_deepseek(
            deepseek_input, DEEPSEEK_PROMPT_1)

        # Parse score
        score = 0
        if "SCORE: 1" in deepseek_output or "CORRECT: Yes" in deepseek_output.upper():
            score = 1

        total_score += score
        topic_scores[topic] += score

        # Display DeepSeek output
        with log_container:
            deepseek_status.success(f"✅ Analysis complete - Score: {score}/1")
            with deepseek_output_box.container():
                st.markdown(deepseek_output)

            # Score indicator
            if score == 1:
                st.success(f"🎯 Result: Correct (Score: {score}/1)")
            else:
                st.error(f"❌ Result: Incorrect (Score: {score}/1)")

            st.markdown("---")
            st.markdown("---")

        all_analyses.append({
            'topic': topic,
            'question_num': question_num,
            'num_pages': num_images,
            'qwen_output': combined_extraction,
            'deepseek_output': deepseek_output,
            'score': score
        })

        progress_bar.progress((idx + 1) / len(valid_questions))

    # Final Comprehensive Analysis
    with log_container:
        st.markdown("## Final Comprehensive Analysis")
        st.markdown("### Step 5: Complete Student Assessment")
        final_status = st.empty()
        final_output_box = st.empty()

    status_text.text("Generating comprehensive feedback...")
    final_status.info("🎓 Analyzing overall performance across all questions...")

    aggregated_input = "=== STUDENT TEST ANALYSIS - ALL RESPONSES ===\n\n"
    for analysis in all_analyses:
        aggregated_input += f"--- {analysis['topic']} - Question {analysis['question_num']} "
        if analysis['num_pages'] > 1:
            aggregated_input += f"({analysis['num_pages']} pages) "
        aggregated_input += "---\n"
        aggregated_input += f"Score: {analysis['score']}/1\n"
        aggregated_input += analysis['deepseek_output']
        aggregated_input += "\n\n" + "="*50 + "\n\n"

    final_feedback = process_with_deepseek(aggregated_input, DEEPSEEK_PROMPT_2)

    with log_container:
        final_status.success("✅ Comprehensive analysis complete!")
        with final_output_box.container():
            st.markdown("#### Aggregated Input to DeepSeek")
            with st.expander("View aggregated data sent to AI", expanded=False):
                st.code(aggregated_input, language="text")

            st.markdown("#### 📝 Final Student Feedback")
            st.info(final_feedback)

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
        'uploaded_images': [],
        'current_test_result': None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ==================== SIDEBAR NAVIGATION ====================


def render_sidebar():
    """Render sidebar navigation for logged-in users"""
    user = st.session_state.user

    with st.sidebar:
        st.markdown("### 👤 User Profile")
        st.markdown(f"**Name:** {user['name']}")
        st.markdown(f"**Grade:** {user['grade']}")
        st.markdown(f"**Age:** {user['age']}")
        st.markdown("---")

        # Navigation buttons
        if st.button("📊 Dashboard", use_container_width=True,
                     type="primary" if st.session_state.page == 'dashboard' else "secondary"):
            st.session_state.page = 'dashboard'
            st.rerun()

        if st.button("📝 Take New Test", use_container_width=True,
                     type="primary" if st.session_state.page == 'test' else "secondary"):
            st.session_state.page = 'test'
            st.session_state.uploaded_images = []
            st.rerun()

        if st.button("🔍 View AI Logs", use_container_width=True,
                     type="primary" if st.session_state.page == 'ai_logs' else "secondary"):
            st.session_state.page = 'ai_logs'
            st.rerun()

        st.markdown("---")

        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ==================== PAGE FUNCTIONS ====================


def login_page():
    """Login page UI"""
    st.title("🎓 AI Math Tutor")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### Login to Your Account")
        st.markdown("---")

        with st.form("login_form"):
            username = st.text_input(
                "Username", placeholder="Enter your username")
            password = st.text_input(
                "Password", type="password", placeholder="Enter your password")

            col_a, col_b = st.columns(2)

            with col_a:
                submitted = st.form_submit_button(
                    "Login", use_container_width=True)

            with col_b:
                signup_btn = st.form_submit_button(
                    "Sign Up", use_container_width=True)

            if submitted:
                if username and password:
                    success, user_data = verify_user(username, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user = user_data
                        st.session_state.page = 'dashboard'
                        st.success("Login successful!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid username or password!")
                else:
                    st.warning("Please fill in all fields!")

            if signup_btn:
                st.session_state.page = 'signup'
                st.rerun()


def signup_page():
    """Sign up page UI"""
    st.title("🎓 AI Math Tutor - Professional Edition")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### Student Registration")
        st.markdown("---")

        with st.form("signup_form"):
            st.markdown("**Account Credentials**")
            username = st.text_input(
                "Username", placeholder="Choose a unique username")
            password = st.text_input(
                "Password", type="password", placeholder="Min 6 characters")
            confirm_password = st.text_input(
                "Confirm Password", type="password", placeholder="Re-enter password")

            st.markdown("---")
            st.markdown("**Student Information**")

            name = st.text_input(
                "Full Name", placeholder="Enter your full name")

            col_x, col_y = st.columns(2)
            with col_x:
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            with col_y:
                age = st.number_input(
                    "Age", min_value=13, max_value=25, value=16)

            grade = st.selectbox("Grade Level", ["O Level", "A Level"])

            st.markdown("---")

            col_a, col_b = st.columns(2)

            with col_a:
                submitted = st.form_submit_button(
                    "Create Account", use_container_width=True)

            with col_b:
                back_btn = st.form_submit_button(
                    "Back to Login", use_container_width=True)

            if submitted:
                if not all([username, password, confirm_password, name]):
                    st.error("Please fill in all fields!")
                elif len(username) < 3:
                    st.error("Username must be at least 3 characters!")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters!")
                elif password != confirm_password:
                    st.error("Passwords do not match!")
                else:
                    success, message = create_user(
                        username, password, name, gender, grade, age)
                    if success:
                        st.success(f"{message}")
                        st.info("Please login with your credentials.")
                        time.sleep(2)
                        st.session_state.page = 'login'
                        st.rerun()
                    else:
                        st.error(f"{message}")

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
            overall = feedback_text.split('OVERALL SUMMARY:')[
                1].split('STRONG TOPICS:')[0].strip()
            sections['overall'] = overall

        if 'STRONG TOPICS:' in feedback_text:
            strong = feedback_text.split('STRONG TOPICS:')[
                1].split('WEAK TOPICS:')[0].strip()
            sections['strong_topics'] = [line.strip(
                '- ').strip() for line in strong.split('\n') if line.strip().startswith('-')]

        if 'WEAK TOPICS:' in feedback_text:
            weak = feedback_text.split('WEAK TOPICS:')[1].split(
                'RECOMMENDATIONS:')[0].strip()
            sections['weak_topics'] = [line.strip(
                '- ').strip() for line in weak.split('\n') if line.strip().startswith('-')]

        if 'RECOMMENDATIONS:' in feedback_text:
            recommendations = feedback_text.split('RECOMMENDATIONS:')[
                1].split('ENCOURAGEMENT:')[0].strip()
            sections['recommendations'] = recommendations

        if 'ENCOURAGEMENT:' in feedback_text:
            encouragement = feedback_text.split('ENCOURAGEMENT:')[1].strip()
            sections['encouragement'] = encouragement

    except Exception as e:
        sections['overall'] = feedback_text

    return sections


def dashboard_page():
    """Main dashboard page with statistics and graphs"""
    user = st.session_state.user

    # Render sidebar
    render_sidebar()

    # Main content
    st.title("📊 Progress Dashboard")

    # Get user's test history
    tests = get_user_tests(user['username'])

    if not tests:
        st.info("Welcome! You haven't taken any tests yet.")
        st.markdown("### Get Started")
        st.markdown(
            "Click **'Take New Test'** in the sidebar to begin your first mathematics assessment.")

        st.markdown("---")
        st.markdown("### Topics Covered in Each Test")
        st.caption("Each test contains 10 questions (2 per topic)")

        cols = st.columns(3)
        for idx, topic in enumerate(MATH_TOPICS):
            with cols[idx % 3]:
                st.markdown(f"**{topic}**")
                st.caption(TOPIC_DESCRIPTIONS[topic])

    else:
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Tests", len(tests))

        with col2:
            avg_score = sum(t.get('total_score', 0)
                            for t in tests) / len(tests)
            st.metric("Average Score", f"{avg_score:.1f}/10")

        with col3:
            latest_score = tests[0].get('total_score', 0)
            st.metric("Latest Score", f"{latest_score}/10")

        with col4:
            avg_percentage = (avg_score / 10) * 100
            st.metric("Average %", f"{avg_percentage:.0f}%")

        st.markdown("---")

        # Progress over time
        if len(tests) > 1:
            st.markdown("### Score Progress Over Time")

            # Safe date parsing with error handling
            dates = []
            scores = []
            for t in reversed(tests):
                try:
                    timestamp = t.get('timestamp', '')
                    if timestamp:
                        # Handle both ISO format and other formats
                        if isinstance(timestamp, str):
                            dt = datetime.fromisoformat(
                                timestamp.replace('Z', '+00:00'))
                        else:
                            dt = timestamp
                        dates.append(dt.strftime('%b %d'))
                    else:
                        dates.append('Unknown')
                    
                    # Add score
                    scores.append(t.get('total_score', 0))
                except Exception as e:
                    print(f"Error parsing date: {e}")
                    dates.append('Unknown')
                    scores.append(0)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=scores,
                mode='lines+markers',
                name='Score',
                line=dict(color='#3498DB', width=3),
                marker=dict(size=12, color='#3498DB'),
                fill='tozeroy',
                fillcolor='rgba(52, 152, 219, 0.1)'
            ))

            fig.update_layout(
                xaxis_title="Test Date",
                yaxis_title="Score (out of 10)",
                yaxis=dict(range=[0, 11]),
                height=400,
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

        # Topic-wise performance
        st.markdown("### Topic-wise Performance")

        topic_scores = {topic: [] for topic in MATH_TOPICS}
        for test in tests:
            topics = test.get('topic_scores', {})
            for topic, score in topics.items():
                if topic in topic_scores:
                    topic_scores[topic].append(score)

        topic_averages = {
            topic: (sum(scores) / len(scores)) if scores else 0
            for topic, scores in topic_scores.items()
        }

        colors = ['#E74C3C' if v < 1 else '#F39C12' if v < 1.5 else '#27AE60'
                  for v in topic_averages.values()]

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

        # AI Analysis
        st.markdown("---")
        st.markdown("### 🤖 AI Tutor Analysis")

        latest_test = tests[0]
        if latest_test.get('final_feedback'):
            ai_analysis = parse_ai_feedback(latest_test['final_feedback'])

            if ai_analysis['overall']:
                st.markdown("#### Overall Performance")
                st.info(ai_analysis['overall'])

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 💪 Strong Areas")
                if ai_analysis['strong_topics']:
                    for topic in ai_analysis['strong_topics']:
                        st.success(f"✅ {topic}")
                else:
                    st.info("Keep practicing to build your strengths!")

            with col2:
                st.markdown("#### 📈 Areas to Improve")
                if ai_analysis['weak_topics']:
                    for topic in ai_analysis['weak_topics']:
                        st.warning(f"⚠️ {topic}")
                else:
                    st.success("All topics are strong!")

            if ai_analysis['recommendations']:
                st.markdown("#### 📚 Study Recommendations")
                st.markdown(ai_analysis['recommendations'])

            if ai_analysis['encouragement']:
                st.markdown("#### 💬 Message from Your AI Tutor")
                st.success(ai_analysis['encouragement'])
        else:
            st.warning("AI analysis not available for latest test.")

        # Recent test results
        st.markdown("---")
        st.markdown("### 📝 Recent Test Results")

        for idx, test in enumerate(tests[:3], 1):
            try:
                test_date = datetime.fromisoformat(
                    test['timestamp']).strftime('%B %d, %Y at %H:%M')
            except:
                test_date = "Unknown date"
            
            score = test.get('total_score', 0)
            percentage = (score / 10) * 100

            with st.expander(f"Test #{idx} - {test_date} - Score: {score}/10 ({percentage:.0f}%)"):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown("**Topic Scores:**")
                    for topic in MATH_TOPICS:
                        topic_score = test.get(
                            'topic_scores', {}).get(topic, 0)
                        status = "✅ Excellent" if topic_score == 2 else "👍 Good" if topic_score == 1 else "📚 Needs Improvement"
                        st.write(f"{topic}: {topic_score}/2 - {status}")

                with col2:
                    st.metric("Total Score", f"{score}/10")
                    st.metric("Percentage", f"{percentage:.0f}%")

                if test.get('final_feedback'):
                    st.markdown("---")
                    st.markdown("**Complete AI Feedback:**")
                    st.info(test['final_feedback'])


def test_page():
    """Test taking page with collective upload at the end"""
    user = st.session_state.user

    # Render sidebar
    render_sidebar()

    # Main content
    st.title("📝 Mathematics Assessment")

    st.info("""
    **Test Instructions:**
    - This test contains **10 questions** (2 from each topic)
    - Review all questions below
    - Write your solutions on paper with **question numbers clearly marked** (e.g., "Question 1", "Q1", etc.)
    - For multi-page solutions, write the question number on the first page
    - Upload **ALL solution images together** at the bottom
    - AI will automatically detect which image belongs to which question
    - Click **Submit Test** when all images are uploaded
    """)

    st.markdown("---")

    # Display all questions organized by topic
    question_counter = 1
    for topic in MATH_TOPICS:
        st.markdown(f"### 📚 {topic}")
        st.caption(TOPIC_DESCRIPTIONS[topic])

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Question {question_counter}:**")
            st.info(MATH_QUESTIONS[topic]["question_1"])
            question_counter += 1

        with col2:
            st.markdown(f"**Question {question_counter}:**")
            st.info(MATH_QUESTIONS[topic]["question_2"])
            question_counter += 1

        st.markdown("---")

    # Collective upload section
    st.markdown("## 📤 Upload All Solutions")
    st.markdown("### Upload all your solution images below")

    st.warning("""
    **IMPORTANT:** 
    - Make sure each solution has its question number written clearly
    - For multi-page solutions, write the question number on the first page
    - Upload images in any order - AI will organize them automatically
    """)

    uploaded_files = st.file_uploader(
        "Select all solution images (you can upload multiple at once)",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        key="collective_upload"
    )

    if uploaded_files:
        st.session_state.uploaded_images = [
            file.read() for file in uploaded_files]
        st.success(f"✅ Uploaded {len(uploaded_files)} image(s)")

        # Show preview of uploaded images
        st.markdown("### 👁️ Preview of Uploaded Images")
        cols = st.columns(min(4, len(uploaded_files)))
        for idx, img_bytes in enumerate(st.session_state.uploaded_images):
            with cols[idx % 4]:
                st.caption(f"Image {idx + 1}")
                image = Image.open(io.BytesIO(img_bytes))
                st.image(image, use_column_width=True)
    else:
        st.session_state.uploaded_images = []

    st.markdown("---")

    # Submit button
    st.markdown("### Ready to Submit?")

    uploaded_count = len(st.session_state.uploaded_images)
    st.write(f"**Total images uploaded: {uploaded_count}**")

    if uploaded_count == 0:
        st.warning("Please upload at least one solution image.")
    elif uploaded_count < 10:
        st.warning(
            f"You've uploaded {uploaded_count} images. Typically, you should have at least 10 images (one per question). Continue if you have multi-page solutions.")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        if st.button("🚀 Submit Test for AI Analysis", use_container_width=True, type="primary", disabled=(uploaded_count == 0)):
            if uploaded_count > 0:
                # Navigate to AI logs page and start processing
                st.session_state.page = 'ai_logs'
                st.session_state.processing_test = True
                st.rerun()


def ai_logs_page():
    """AI Logs page showing real-time processing with detection"""
    user = st.session_state.user

    # Render sidebar
    render_sidebar()

    # Main content
    st.title("🔍 AI Processing Logs")
    st.markdown("Real-time view of AI pipeline processing your test")
    st.markdown("---")

    # Check if we need to process a test
    if st.session_state.get('processing_test', False):
        # Create container for logs
        log_container = st.container()

        with log_container:
            st.info("🚀 Starting AI analysis pipeline...")
            st.markdown("---")

        # Get uploaded images
        uploaded_images = st.session_state.uploaded_images

        if not uploaded_images:
            st.error("No images found to process!")
            st.session_state.processing_test = False
            return

        # Step 1: Detect and group images
        grouped_questions, detected_images = detect_and_group_images(
            uploaded_images, log_container)

        # Step 2: Map to test structure
        test_questions = map_to_test_structure(
            grouped_questions, log_container)

        # Check if we have enough questions
        valid_questions = [q for q in test_questions if q['images']]

        with log_container:
            if len(valid_questions) < 10:
                st.warning(
                    f"⚠️ Warning: Only {len(valid_questions)}/10 questions were successfully mapped. Proceeding with available questions...")

        # Step 3: Run analysis with streaming
        result = analyze_test_images_with_streaming(
            test_questions, log_container)

        # Save result
        save_test_result(user['username'], result)

        # Store in session state
        st.session_state.current_test_result = result
        st.session_state.processing_test = False

        # Show completion message
        with log_container:
            st.markdown("---")
            st.success(
                "🎉 Test analysis complete! Results saved to your dashboard.")
            st.balloons()

        # Clear uploaded images
        st.session_state.uploaded_images = []

    else:
        # Show most recent test logs if available
        tests = get_user_tests(user['username'])

        if not tests:
            st.info(
                "No test results available yet. Take a test to see AI processing logs.")
        else:
            st.markdown("### Most Recent Test Analysis")

            latest_test = tests[0]
            try:
                test_date = datetime.fromisoformat(
                    latest_test['timestamp']).strftime('%B %d, %Y at %H:%M')
            except:
                test_date = "Unknown date"
            
            st.caption(f"Test Date: {test_date}")

            st.markdown("---")

            # Display individual analyses
            if 'individual_analyses' in latest_test:
                for idx, analysis in enumerate(latest_test['individual_analyses'], 1):
                    st.markdown(
                        f"### Question {idx}: {analysis['topic']} - Q{analysis['question_num']}")
                    st.markdown(f"**Pages uploaded:** {analysis['num_pages']}")
                    st.markdown("---")

                    st.markdown("#### Step: Qwen-VL Text Extraction")
                    st.success(
                        f"✅ Extraction complete ({analysis['num_pages']} page(s))")
                    st.code(analysis['qwen_output'], language="text")

                    st.markdown("#### Step: DeepSeek Analysis & Scoring")
                    st.success(
                        f"✅ Analysis complete - Score: {analysis['score']}/1")
                    st.markdown(analysis['deepseek_output'])

                    if analysis['score'] == 1:
                        st.success(
                            f"🎯 Result: Correct (Score: {analysis['score']}/1)")
                    else:
                        st.error(
                            f"❌ Result: Incorrect (Score: {analysis['score']}/1)")

                    st.markdown("---")
                    st.markdown("---")

            # Final analysis
            st.markdown("## Final Comprehensive Analysis")
            st.markdown("### Complete Student Assessment")

            if 'aggregated_input' in latest_test:
                st.markdown("#### Aggregated Input to DeepSeek")
                with st.expander("View aggregated data sent to AI", expanded=False):
                    st.code(latest_test['aggregated_input'], language="text")

            st.markdown("#### 📝 Final Student Feedback")
            if 'final_feedback' in latest_test:
                st.info(latest_test['final_feedback'])


# ==================== MAIN APP ====================

def main():
    """Main application router"""
    init_session_state()

    if not st.session_state.logged_in:
        if st.session_state.page == 'signup':
            signup_page()
        else:
            login_page()
    else:
        if st.session_state.page == 'dashboard':
            dashboard_page()
        elif st.session_state.page == 'test':
            test_page()
        elif st.session_state.page == 'ai_logs':
            ai_logs_page()
        else:
            dashboard_page()


if __name__ == "__main__":
    main()