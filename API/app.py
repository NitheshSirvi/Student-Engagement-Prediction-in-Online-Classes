import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
import urllib.parse
from datetime import datetime


import cv2 # ADD THIS AT THE VERY TOP WITH YOUR OTHER IMPORTS


# 1. SETUP PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models", "saved_models")
sys.path.append(BASE_DIR)

try:
    from database.db_logger import log_prediction_to_sql
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

# ---------------------------------------------------------
# 2. APP CONFIGURATION & ASSET LOADING
# ---------------------------------------------------------
st.set_page_config(page_title="Insight | Student Engagement", page_icon="🎓", layout="wide")

@st.cache_resource
def load_assets():
    model_path = os.path.join(MODEL_SAVE_DIR, "xgboost_model.pkl")
    scaler_path = os.path.join(MODEL_SAVE_DIR, "scaler.pkl")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception:
        return None, None

model, scaler = load_assets()

if model is None or scaler is None:
    st.error("⚠️ Error: The AI model or scaler could not be found.")
    st.info("Run `python main.py` in your terminal to generate the models.")
    st.stop()

# Initialize session state for cross-tab data sharing
if 'class_data' not in st.session_state:
    st.session_state['class_data'] = None

# ---------------------------------------------------------
# 3. UI HEADER
# ---------------------------------------------------------
st.title("🎓 Insight: Active Learning Monitor")
st.markdown("Real-time predictive analytics to prevent student drop-offs and monitor class health.")
st.divider()

# Added a 5th Tab for Advanced Analytics
tab1, tab2, tab3, tab4, tab5,tab6= st.tabs([
    "🔴 Live Student Desk", 
    "📂 Batch Class Analysis", 
    "📊 Student Profiler", 
    "⚙️ AI Model Insights",
    "📈 Advanced Analytics",# NEW REAL-WORLD FEATURE
    "📷 Live Proctoring" # NEW TAB ADDED HERE
])

# ---------------------------------------------------------
# 4. SIDEBAR: LIVE DATA FEED
# ---------------------------------------------------------
st.sidebar.header("📡 Live Student Feed")
st.sidebar.caption("Adjust sliders to simulate live telemetry.")

student_id = st.sidebar.text_input("Student ID", value="STU-8042")
time_spent = st.sidebar.slider("Time on Platform (mins)", 0, 300, 45)
clicks = st.sidebar.slider("Video Clicks", 0, 50, 12)
quiz_score = st.sidebar.slider("Last Quiz Score (%)", 0, 100, 75)
forum_posts = st.sidebar.slider("Forum Posts Today", 0, 10, 1)

input_df = pd.DataFrame([[time_spent, clicks, quiz_score, forum_posts]], 
                        columns=['time_on_platform_mins', 'num_video_clicks', 'quiz_score_avg', 'forum_posts'])
input_scaled = scaler.transform(input_df)

prediction = model.predict(input_scaled)[0]
prob = model.predict_proba(input_scaled)[0][1] 

# ---------------------------------------------------------
# TAB 1: LIVE PREDICTION DESK (WITH ONE-CLICK EMAIL)
# ---------------------------------------------------------
with tab1:
    st.subheader(f"Current Status: {student_id}")
    col1, col2, col3 = st.columns([1.2, 1, 1])
    
    with col1:
        st.markdown("### Model Verdict")
        if prediction == 1:
            st.success("✅ **Actively Engaged**")
            action = "Student is on track. Keep monitoring."
        else:
            st.error("⚠️ **At-Risk / Disengaged**")
            
            # Simple, clean AI recommendation text
            if time_spent > 120 and clicks < 5:
                action = "💡 Action: Student might be stuck. Send a direct message offering help."
            elif quiz_score < 50:
                action = "💡 Action: Recommend prerequisite reading material."
            else:
                action = "💡 Action: Launch a quick live poll to regain attention."
        
            st.info(action)
            
            # --- NEW UNIQUE FEATURE: AUTOMATED MICRO-INTERVENTION ---
            st.markdown("---")
            st.markdown("### 🤖 Automated System Action")
            st.warning("Confidence score dropped below threshold. Triggering auto-intervention...")
            
            # Simulate sending a payload to the student's screen
            if st.button("👁️ View What The Student Sees", key="view_intervention_btn"):
                st.toast("✅ Intervention sent successfully to student's screen!", icon="🚀")
                
                # A simulated pop-up of what appears on the student's laptop
                st.info("""
                **[Simulated Pop-Up on Student's Screen]**
                
                🔔 *Wait! Before you move on...*
                **Quick Check-In:** You've been on this module for a while. What is the most confusing part so far?
                * [ ] The math formulas
                * [ ] The coding syntax
                * [ ] I'm just taking a quick break
                
                *(Submitting this form pauses the lecture timer)*
                """)
            # --------------------------------------------------------

        with st.expander("Save to Database"):
            st.write("Log this incident for end-of-semester reporting.")
            
            # FIXED: Added a unique key to the button so Streamlit doesn't get confused
            if st.button("💾 Log Incident", key="log_btn_tab1"):
                if DB_AVAILABLE:
                    success = log_prediction_to_sql(time_spent, clicks, quiz_score, forum_posts, prediction, prob)
                    if success:
                        st.success("Saved to SQL Server!")
                    else:
                        st.warning("SQL connection failed. Ensure database is running.")
                else:
                    st.warning("Database module not found. Logged locally.")
            # ---------------------------------------------------
            # --- NEW REAL-WORLD FEATURE: Automated Email Alert ---
            subject = urllib.parse.quote("Checking in on your class progress")
            body = urllib.parse.quote(f"Hi {student_id},\n\nI noticed from your recent dashboard metrics that you might be facing some challenges with the current module. \n\nAre you available for a quick 5-minute chat so I can help clarify the topics?\n\nBest regards,\nYour Instructor")
            mailto_link = f"mailto:{student_id.lower()}@university.edu?subject={subject}&body={body}"
            
            st.markdown(f'''
                <a href="{mailto_link}" target="_blank">
                    <button style="background-color:#ff4b4b; color:white; border:none; padding:10px 20px; border-radius:5px; cursor:pointer; font-weight:bold; width:100%;">
                        ✉️ Click to Email Student Automatically
                    </button>
                </a>
                <br><br>
            ''', unsafe_allow_html=True)
            # ---------------------------------------------------

        with st.expander("Save to Database"):
            if st.button("💾 Log Incident"):
                if DB_AVAILABLE:
                    success = log_prediction_to_sql(time_spent, clicks, quiz_score, forum_posts, prediction, prob)
                    if success: st.success("Saved to SQL Server!")
                    else: st.warning("SQL connection failed.")
                else:
                    st.warning("Database module not found.")

    with col2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100, title={'text': "Engagement Confidence %"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 40], 'color': "#ff4b4b"}, {'range': [40, 75], 'color': "#faca2b"}, {'range': [75, 100], 'color': "#00cc96"}]}
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col3:
        st.markdown("### Live Telemetry")
        st.metric(label="Platform Time", value=f"{time_spent} m", delta=f"{time_spent - 60} m from avg")
        st.metric(label="Interactions", value=clicks, delta=f"{clicks - 15} from avg")
        st.metric(label="Quiz Average", value=f"{quiz_score}%")

# ---------------------------------------------------------
# TAB 2: BATCH CLASS ANALYSIS (SMART CSV)
# ---------------------------------------------------------
with tab2:
    st.subheader("📂 Upload Class Data for Batch Prediction")
    uploaded_file = st.file_uploader("Upload Student CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            numeric_cols = batch_df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) < 4:
                st.error("⚠️ Your CSV needs at least 4 numerical columns to make a prediction.")
            else:
                st.info("💡 Map your CSV columns to the AI's required inputs:")
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1: time_col = st.selectbox("Time Spent", numeric_cols, index=0)
                with col_m2: click_col = st.selectbox("Video Clicks", numeric_cols, index=min(1, len(numeric_cols)-1))
                with col_m3: quiz_col = st.selectbox("Quiz Score", numeric_cols, index=min(2, len(numeric_cols)-1))
                with col_m4: forum_col = st.selectbox("Forum Posts", numeric_cols, index=min(3, len(numeric_cols)-1))
                
                if st.button("Map Data & Predict"):
                    with st.spinner("Processing data..."):
                        features_df = batch_df[[time_col, click_col, quiz_col, forum_col]].copy().fillna(0)
                        features_df.columns = ['time_on_platform_mins', 'num_video_clicks', 'quiz_score_avg', 'forum_posts']
                        
                        # Save mapped data to memory for Tab 5 Analytics
                        st.session_state['class_data'] = features_df.copy()
                        
                        batch_scaled = scaler.transform(features_df)
                        batch_preds = model.predict(batch_scaled)
                        batch_probs = model.predict_proba(batch_scaled)[:, 1]
                        
                        results_df = batch_df.copy()
                        results_df['Engagement_Score(%)'] = np.round(batch_probs * 100, 2)
                        results_df['Status'] = ["Engaged" if p == 1 else "At-Risk" for p in batch_preds]
                        
                        st.session_state['class_data']['Status'] = results_df['Status'] # Add status for coloring
                        
                        engaged_count = int(sum(batch_preds == 1))
                        at_risk_count = int(sum(batch_preds == 0))
                        
                        colA, colB = st.columns([1, 2])
                        with colA:
                            st.metric("Total Students Processed", len(batch_preds))
                            st.metric("Engaged", engaged_count, delta="Healthy")
                            st.metric("At-Risk", at_risk_count, delta="Requires Attention", delta_color="inverse")
                        with colB:
                            pie_data = pd.DataFrame({'Status': ['Engaged', 'At-Risk'], 'Students': [engaged_count, at_risk_count]})
                            fig_pie = px.pie(pie_data, names='Status', values='Students', color='Status',
                                             color_discrete_map={'Engaged': '#00cc96', 'At-Risk': '#ff4b4b'}, hole=0.3)
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        st.dataframe(results_df)
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Processed Report", data=csv, file_name='class_report.csv', mime='text/csv')
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ---------------------------------------------------------
# TAB 3 & 4: PROFILER & AI INSIGHTS
# ---------------------------------------------------------
with tab3:
    st.subheader(f"Behavioral Profiling: {student_id}")
    categories = ['Time Spent', 'Video Clicks', 'Quiz Score', 'Forum Activity']
    student_stats = [min((time_spent / 150) * 100, 100), min((clicks / 30) * 100, 100), quiz_score, min((forum_posts / 5) * 100, 100)]
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=student_stats, theta=categories, fill='toself', name=f'Student {student_id}'))
    fig_radar.add_trace(go.Scatterpolar(r=[60, 50, 70, 40], theta=categories, fill='none', name='Class Average', line_color='gray', line_dash='dash'))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=400)
    st.plotly_chart(fig_radar, use_container_width=True)

with tab4:
    st.subheader("AI Decision Transparency")
    try:
        importance = model.feature_importances_
        df_imp = pd.DataFrame({'Feature': ['Time on Platform', 'Video Clicks', 'Quiz Score', 'Forum Posts'], 'Importance': importance}).sort_values(by='Importance', ascending=True)
        fig_bar = px.bar(df_imp, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='teal')
        st.plotly_chart(fig_bar, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating chart: {str(e)}")

# ---------------------------------------------------------
# TAB 5: ADVANCED ANALYTICS (NEW REAL-WORLD FEATURE)
# ---------------------------------------------------------
with tab5:
    st.subheader("📈 Advanced Data Correlation")
    st.write("Discover hidden patterns in your students' behavior.")
    
    # Check if the user has uploaded and mapped data in Tab 2
    if st.session_state['class_data'] is not None:
        analysis_df = st.session_state['class_data']
        
        col_x, col_y = st.columns(2)
        
        with col_x:
            st.markdown("**Behavioral Correlation Heatmap**")
            st.caption("Values closer to 1.0 show a strong positive relationship.")
            
            
            # Drop the Status column for pure math correlation
            corr_df = analysis_df.drop(columns=['Status'], errors='ignore')
            corr_matrix = corr_df.corr()
            
            fig_heat = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
            fig_heat.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_heat, use_container_width=True)
            
        with col_y:
            st.markdown("**Time vs. Performance Distribution**")
            st.caption("How does time spent affect quiz scores?")
            
            fig_scatter = px.scatter(
                analysis_df, 
                x="time_on_platform_mins", 
                y="quiz_score_avg", 
                color="Status",
                color_discrete_map={'Engaged': '#00cc96', 'At-Risk': '#ff4b4b'},
                labels={"time_on_platform_mins": "Time on Platform (Mins)", "quiz_score_avg": "Quiz Score (%)"}
            )
            fig_scatter.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_scatter, use_container_width=True)
            
    else:
        # Beautiful empty state if no data is uploaded yet
        st.info("👈 **No Class Data Found.** Please upload and process a CSV file in **Tab 2: Batch Class Analysis** to unlock these advanced mathematical visualizations.")

        # ---------------------------------------------------------
# TAB 6: LIVE PROCTORING & COMPUTER VISION (WEBAPP MERGE)
# ---------------------------------------------------------
# ---------------------------------------------------------
# TAB 6: LIVE PROCTORING & COMPUTER VISION (WEBAPP MERGE)
# ---------------------------------------------------------
with tab6:
    st.subheader("📷 Live Video Engagement Tracking")
    st.write("Using OpenCV to detect head posture and screen focus in real-time.")
    
    col_video, col_metrics = st.columns([2, 1])
    
    with col_video:
        # A checkbox to start/stop the camera safely
        run_video = st.checkbox("🟢 Start Webcam Feed")
        
        # This is where the video will be displayed on the webpage
        FRAME_WINDOW = st.image([]) 
        
    with col_metrics:
        st.markdown("### Vision AI Status")
        status_text = st.empty() 
        
        
    if run_video:
        # 1. Open the webcam (0 is usually the default laptop camera)
        camera = cv2.VideoCapture(0)
        
        # 2. Load OpenCV's built-in face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        while run_video:
            # 3. Read the video frame
            success, frame = camera.read()
            if not success:
                st.error("Could not access webcam.")
                break
                
            # Streamlit needs RGB colors, but OpenCV uses BGR. We convert it:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # 4. Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
            
            # 5. Logic: If looking at screen (face detected) vs Looking away
            if len(faces) > 0:
                status = "✅ ENGAGED: Student is focused on screen."
                status_text.success(status)
                
                # Draw a green box around the face
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(frame, "Focused", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                status = "⚠️ DISTRACTED: Student looked away or left."
                status_text.error(status)
                
                # Add a red warning border to the video frame
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), 10)
                cv2.putText(frame, "DISTRACTED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

            # 6. Push the frame to the Streamlit web page
            FRAME_WINDOW.image(frame)
            
    else:
        # Turn off the camera when the checkbox is unchecked
        try:
            camera.release()
        except:
            pass
        st.info("Webcam is currently off. Check the box to start tracking.")

   
    
    