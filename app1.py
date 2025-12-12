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
    page_icon="",
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


def analyze_all_images_simple(all_image_bytes, log_container):
    """
    Simplified pipeline:
    1. Extract text from ALL images with Qwen
    2. Send everything to DeepSeek 1 (it handles question matching)
    3. Send DeepSeek 1 output to DeepSeek 2
    """

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    with log_container:
        st.markdown("##  Step 1: Qwen-VL Text Extraction from All Images")
        st.info(
            f"Extracting text from {len(all_image_bytes)} uploaded image(s)...")
        st.markdown("---")

    # Step 1: Extract text from all images
    extracted_texts = []
    for idx, image_bytes in enumerate(all_image_bytes, 1):
        status_text.text(
            f"Extracting text from image {idx}/{len(all_image_bytes)}...")

        with log_container:
            st.markdown(f"###  Processing Image {idx}/{len(all_image_bytes)}")
            extraction_status = st.empty()
            extraction_output = st.empty()

        extraction_status.info(f"🔄 Extracting text from image {idx}...")

        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        qwen_output = process_image_with_qwen(image_base64, QWEN_PROMPT)
        extracted_texts.append(f"=== IMAGE {idx} ===\n{qwen_output}\n")

        with log_container:
            extraction_status.success(f"✅ Extraction complete for image {idx}")
            with extraction_output.container():
                st.code(qwen_output, language="text")
            st.markdown("---")

        progress_bar.progress(idx / (len(all_image_bytes) + 2))

    combined_extraction = "\n\n".join(extracted_texts)

    with log_container:
        st.markdown("### ✅ All Extractions Complete")
        st.success(
            f"Successfully extracted text from {len(all_image_bytes)} images")
        with st.expander(" View Combined Extracted Text", expanded=False):
            st.code(combined_extraction, language="text")
        st.markdown("---")
        st.markdown("---")

    # Step 2: Prepare all questions
    status_text.text("Preparing questions for analysis...")

    all_questions_text = ""
    question_counter = 1
    for topic in MATH_TOPICS:
        all_questions_text += f"\nQuestion {question_counter} ({topic}):\n{MATH_QUESTIONS[topic]['question_1']}\n"
        question_counter += 1
        all_questions_text += f"\nQuestion {question_counter} ({topic}):\n{MATH_QUESTIONS[topic]['question_2']}\n"
        question_counter += 1

    deepseek1_input = f"""TEST QUESTIONS (10 questions total):
{all_questions_text}

STUDENT'S UPLOADED PAGES (All extracted text from {len(all_image_bytes)} images):
{combined_extraction}

Instructions: Please analyze the student's work by matching their solutions to the appropriate questions above. The student may have uploaded pages in any order, and some questions might span multiple pages."""

    # Step 3: DeepSeek 1 Analysis
    with log_container:
        st.markdown("## 🤖 Step 2: DeepSeek Analysis & Grading")
        st.info("Analyzing student's solutions and matching them to questions...")
        deepseek1_status = st.empty()
        deepseek1_output_box = st.empty()

    status_text.text("Running DeepSeek analysis (this may take a moment)...")
    deepseek1_status.info("🔄 DeepSeek is analyzing all solutions...")

    progress_bar.progress((len(all_image_bytes) + 1) /
                          (len(all_image_bytes) + 2))

    deepseek1_output = process_with_deepseek(
        deepseek1_input, DEEPSEEK_PROMPT_1)

    with log_container:
        deepseek1_status.success("✅ DeepSeek analysis complete")
        with deepseek1_output_box.container():
            st.markdown("### 📊 Detailed Question-by-Question Analysis")
            st.markdown(deepseek1_output)

        st.markdown("---")
        st.markdown("---")

    # Step 4: DeepSeek 2 Final Report
    with log_container:
        st.markdown("## 📋 Step 3: Comprehensive Final Report")
        st.info("Generating comprehensive feedback and study recommendations...")
        deepseek2_status = st.empty()
        deepseek2_output_box = st.empty()

    status_text.text("Generating final comprehensive report...")
    deepseek2_status.info("🔄 Creating final assessment report...")

    deepseek2_output = process_with_deepseek(
        deepseek1_output, DEEPSEEK_PROMPT_2)

    progress_bar.progress(1.0)

    with log_container:
        deepseek2_status.success("✅ Final report generated")
        with deepseek2_output_box.container():
            st.markdown("### 🎓 Final Student Assessment")
            st.info(deepseek2_output)

        st.markdown("---")

    # Try to parse score from DeepSeek output (basic parsing)
    total_score = 0
    try:
        # Look for patterns like "Final Average Score: 3.5/5" or "Average: 3.5"
        import re
        score_patterns = [
            r'Final Average Score[:\s]+(\d+\.?\d*)',
            r'Average[:\s]+(\d+\.?\d*)',
            r'Total.*?(\d+)/10',
            r'Score.*?(\d+)/10'
        ]
        for pattern in score_patterns:
            match = re.search(pattern, deepseek1_output)
            if match:
                score_val = float(match.group(1))
                # Convert to 0-10 scale if needed
                if score_val <= 5:
                    total_score = int(score_val * 2)  # Convert 0-5 to 0-10
                else:
                    total_score = int(score_val)
                break
    except:
        total_score = 0

    progress_bar.empty()
    status_text.empty()

    return {
        'total_score': total_score,
        'topic_scores': {},  # Not tracking individual topics in simplified version
        'all_extracted_text': combined_extraction,
        'deepseek1_analysis': deepseek1_output,
        'final_feedback': deepseek2_output,
        'number_of_images': len(all_image_bytes)
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
        if st.button(" Dashboard", use_container_width=True,
                     type="primary" if st.session_state.page == 'dashboard' else "secondary"):
            st.session_state.page = 'dashboard'
            st.rerun()

        if st.button(" Take New Test", use_container_width=True,
                     type="primary" if st.session_state.page == 'test' else "secondary"):
            st.session_state.page = 'test'
            st.session_state.uploaded_images = []
            st.rerun()

        if st.button("🤖 View AI Logs", use_container_width=True,
                     type="primary" if st.session_state.page == 'ai_logs' else "secondary"):
            st.session_state.page = 'ai_logs'
            st.rerun()

        st.markdown("---")

        if st.button(" Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ==================== PAGE FUNCTIONS ====================


def login_page():
    """Login page UI"""
    st.title(" AI Math Tutor")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("###  Login to Your Account")
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
    st.title(" AI Math Tutor")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("###  Student Registration")
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
                    st.error("❌ Please fill in all fields!")
                elif len(username) < 3:
                    st.error("❌ Username must be at least 3 characters!")
                elif len(password) < 6:
                    st.error("❌ Password must be at least 6 characters!")
                elif password != confirm_password:
                    st.error("❌ Passwords do not match!")
                else:
                    success, message = create_user(
                        username, password, name, gender, grade, age)
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
        st.info("👋 Welcome! You haven't taken any tests yet.")
        st.markdown("###  Get Started")
        st.markdown(
            "Click **' Take New Test'** in the sidebar to begin your first mathematics assessment.")

        st.markdown("---")
        st.markdown("###  Topics Covered in Each Test")
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
            st.metric(" Total Tests", len(tests))

        with col2:
            avg_score = sum(t.get('total_score', 0)
                            for t in tests) / len(tests)
            st.metric(" Average Score", f"{avg_score:.1f}/10")

        with col3:
            latest_score = tests[0].get('total_score', 0)
            st.metric(" Latest Score", f"{latest_score}/10")

        with col4:
            avg_percentage = (avg_score / 10) * 100
            st.metric(" Average %", f"{avg_percentage:.0f}%")

        st.markdown("---")

        # Progress over time
        if len(tests) > 1:
            st.markdown("###  Score Progress Over Time")

            # Safe date parsing
            dates = []
            scores = []
            for t in reversed(tests):
                try:
                    timestamp = t.get('timestamp', '')
                    if timestamp:
                        if isinstance(timestamp, str):
                            dt = datetime.fromisoformat(
                                timestamp.replace('Z', '+00:00'))
                        else:
                            dt = timestamp
                        dates.append(dt.strftime('%b %d'))
                    else:
                        dates.append('Unknown')
                    scores.append(t.get('total_score', 0))
                except Exception as e:
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

        # AI Analysis
        st.markdown("---")
        st.markdown("### 🤖 AI Tutor Analysis")

        latest_test = tests[0]
        if latest_test.get('final_feedback'):
            # Display the final feedback
            st.info(latest_test['final_feedback'])

            # Try to parse structured feedback if available
            ai_analysis = parse_ai_feedback(latest_test['final_feedback'])

            if ai_analysis['overall']:
                st.markdown("####  Overall Performance")
                st.info(ai_analysis['overall'])

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### ✅ Strong Areas")
                if ai_analysis['strong_topics']:
                    for topic in ai_analysis['strong_topics']:
                        st.success(f"✅ {topic}")
                else:
                    st.info("Keep practicing to build your strengths!")

            with col2:
                st.markdown("#### ⚠️ Areas to Improve")
                if ai_analysis['weak_topics']:
                    for topic in ai_analysis['weak_topics']:
                        st.warning(f"⚠️ {topic}")
                else:
                    st.success("All topics are strong!")

            if ai_analysis['recommendations']:
                st.markdown("####  Study Recommendations")
                st.markdown(ai_analysis['recommendations'])

            if ai_analysis['encouragement']:
                st.markdown("#### 💬 Message from Your AI Tutor")
                st.success(ai_analysis['encouragement'])
        else:
            st.warning("AI analysis not available for latest test.")

        # Recent test results
        st.markdown("---")
        st.markdown("###  Recent Test Results")

        for idx, test in enumerate(tests[:5], 1):
            try:
                test_date = datetime.fromisoformat(
                    test['timestamp']).strftime('%B %d, %Y at %H:%M')
            except:
                test_date = "Unknown date"

            score = test.get('total_score', 0)
            percentage = (score / 10) * 100
            num_images = test.get('number_of_images', 'N/A')

            with st.expander(f"Test #{idx} - {test_date} - Score: {score}/10 ({percentage:.0f}%) - {num_images} images uploaded"):
                col1, col2 = st.columns([3, 1])

                with col1:
                    if test.get('deepseek1_analysis'):
                        st.markdown("** Question Analysis:**")
                        st.markdown(test['deepseek1_analysis'])

                with col2:
                    st.metric("Total Score", f"{score}/10")
                    st.metric("Percentage", f"{percentage:.0f}%")
                    st.metric("Images", num_images)

                if test.get('final_feedback'):
                    st.markdown("---")
                    st.markdown("**🎓 Complete AI Feedback:**")
                    st.info(test['final_feedback'])


def test_page():
    """Test taking page with collective upload at the end"""
    user = st.session_state.user

    # Render sidebar
    render_sidebar()

    # Main content
    st.title(" Mathematics Assessment")

    st.info("""
    ** Test Instructions:**
    - This test contains **10 questions** (2 from each topic)
    - Review all questions below
    - Write your solutions on paper (you can use multiple pages per question)
    - **Important:** Write question numbers clearly (e.g., "Question 1", "Q1", "1)") at the top of each solution
    - Upload **ALL solution images together** at the bottom of this page
    - Click **Submit Test** when all images are uploaded
    - AI will automatically analyze your solutions
    """)

    st.markdown("---")

    # Display all questions organized by topic
    question_counter = 1
    for topic in MATH_TOPICS:
        st.markdown(f"###  {topic}")
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

    # Collective upload section at the end
    st.markdown("##  Upload All Your Solutions")
    st.markdown("### Upload all your solution images below")

    st.warning("""
    **⚠️ IMPORTANT UPLOAD INSTRUCTIONS:**
    - Write question numbers clearly on your pages (e.g., "Question 1", "Q1", "1)")
    - For multi-page solutions, write the question number on the first page
    - Upload all solution images together (select multiple files at once)
    - Supported formats: PNG, JPG, JPEG
    - Make sure images are clear and readable
    """)

    uploaded_files = st.file_uploader(
        " Select all solution images (you can upload multiple files at once)",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        key="collective_upload"
    )

    if uploaded_files:
        st.session_state.uploaded_images = [
            file.read() for file in uploaded_files]
        st.success(f"✅ Uploaded {len(uploaded_files)} image(s)")

        # Show preview of uploaded images
        st.markdown("###  Preview of Uploaded Images")

        # Display images in a grid
        num_cols = 4
        cols = st.columns(num_cols)
        for idx, img_bytes in enumerate(st.session_state.uploaded_images):
            with cols[idx % num_cols]:
                st.caption(f"Image {idx + 1}")
                image = Image.open(io.BytesIO(img_bytes))
                st.image(image, use_column_width=True)
    else:
        st.session_state.uploaded_images = []

    st.markdown("---")

    # Submit button
    st.markdown("###  Ready to Submit?")

    uploaded_count = len(st.session_state.uploaded_images)
    st.write(f"**Total images uploaded: {uploaded_count}**")

    if uploaded_count == 0:
        st.warning("⚠️ Please upload at least one solution image.")
    else:
        st.info(f" Ready to analyze {uploaded_count} image(s)")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        if st.button(" Submit Test for AI Analysis", use_container_width=True, type="primary", disabled=(uploaded_count == 0)):
            if uploaded_count > 0:
                # Navigate to AI logs page and start processing
                st.session_state.page = 'ai_logs'
                st.session_state.processing_test = True
                st.rerun()


def ai_logs_page():
    """AI Logs page showing real-time processing"""
    user = st.session_state.user

    # Render sidebar
    render_sidebar()

    # Main content
    st.title("🤖 AI Processing Logs")
    st.markdown("Real-time view of AI pipeline processing your test")
    st.markdown("---")

    # Check if we need to process a test
    if st.session_state.get('processing_test', False):
        # Create container for logs
        log_container = st.container()

        with log_container:
            st.info(" Starting AI analysis pipeline...")
            st.markdown("---")

        # Get uploaded images
        uploaded_images = st.session_state.uploaded_images

        if not uploaded_images:
            st.error("❌ No images found to process!")
            st.session_state.processing_test = False
            return

        # Run simplified analysis
        result = analyze_all_images_simple(uploaded_images, log_container)

        # Save result to Supabase
        save_data = {
            'total_score': result.get('total_score', 0),
            'topic_scores': result.get('topic_scores', {}),
            'individual_analyses': [],  # Not used in simplified version
            'final_feedback': result.get('final_feedback', ''),
            'aggregated_input': result.get('all_extracted_text', ''),
            'deepseek1_analysis': result.get('deepseek1_analysis', ''),
            'number_of_images': result.get('number_of_images', 0)
        }

        save_test_result(user['username'], save_data)

        # Store in session state
        st.session_state.current_test_result = result
        st.session_state.processing_test = False

        # Show completion message
        with log_container:
            st.markdown("---")
            st.success(
                " Test analysis complete! Results saved to your dashboard.")
            st.balloons()

        # Clear uploaded images
        st.session_state.uploaded_images = []

    else:
        # Show most recent test logs if available
        tests = get_user_tests(user['username'])

        if not tests:
            st.info(
                " No test results available yet. Take a test to see AI processing logs.")
        else:
            st.markdown("###  Most Recent Test Analysis")

            latest_test = tests[0]
            try:
                test_date = datetime.fromisoformat(
                    latest_test['timestamp']).strftime('%B %d, %Y at %H:%M')
            except:
                test_date = "Unknown date"

            st.caption(f" Test Date: {test_date}")
            st.caption(
                f" Images Analyzed: {latest_test.get('number_of_images', 'N/A')}")

            st.markdown("---")

            # Display extracted text
            if latest_test.get('aggregated_input'):
                st.markdown("## 🔍 Step 1: Qwen-VL Text Extraction")
                st.success("✅ Text extraction from all images complete")
                with st.expander(" View All Extracted Text", expanded=False):
                    st.code(latest_test['aggregated_input'], language="text")
                st.markdown("---")

            # Display DeepSeek 1 analysis
            if latest_test.get('deepseek1_analysis'):
                st.markdown("## 🤖 Step 2: DeepSeek Analysis & Grading")
                st.success("✅ Question-by-question analysis complete")
                with st.expander(" View Detailed Analysis", expanded=True):
                    st.markdown(latest_test['deepseek1_analysis'])
                st.markdown("---")

            # Display final feedback
            if latest_test.get('final_feedback'):
                st.markdown("##  Step 3: Comprehensive Final Report")
                st.success("✅ Final assessment report generated")
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
