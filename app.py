import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #667eea;
        --secondary: #764ba2;
        --accent: #f5576c;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f5576c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -2px;
    }
    
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 2rem;
        font-family: 'Courier New', monospace;
    }
    
    /* Card styling */
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .stMetric label {
        color: white !important;
        font-weight: 600;
    }
    
    .stMetric .metric-value {
        color: white !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }
    
    /* Prediction box */
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 20px 60px rgba(245, 87, 108, 0.4);
    }
    
    .prediction-value {
        font-size: 5rem;
        font-weight: 700;
        color: white;
        margin: 0;
        text-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .prediction-label {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 0.5rem;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 2rem 0 1rem 0;
        font-weight: 600;
        font-size: 1.2rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.1rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: rgba(245, 87, 108, 0.1);
        border-left: 4px solid #f5576c;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: rgba(48, 209, 88, 0.1);
        border-left: 4px solid #30D158;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load and prepare model
@st.cache_resource
def load_model():
    """Load and train the model"""
    try:
        # Load dataset
        df = pd.read_csv("student_data.csv")
        
        # Encode categorical columns
        le = LabelEncoder()
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = le.fit_transform(df[col])
        
        # Prepare features and target
        X = df.drop("G3", axis=1)
        y = df["G3"]
        
        # Split data
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        
        # Train model
        model = GradientBoostingRegressor(random_state=42)
        model.fit(x_train_scaled, y_train)
        
        # Calculate accuracy
        y_pred = model.predict(x_test_scaled)
        accuracy = r2_score(y_test, y_pred)
        
        return model, scaler, X.columns.tolist(), accuracy, df
    
    except FileNotFoundError:
        st.error("⚠️ student_data.csv file not found! Please upload the dataset.")
        return None, None, None, None, None

# Initialize
model, scaler, feature_names, model_accuracy, df = load_model()

# Header
st.markdown('<h1 class="main-header">🎓 Student Performance Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Academic Analytics</p>', unsafe_allow_html=True)

if model is not None:
    # Sidebar for model info
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/student-male.png", width=80)
        st.title("📊 Model Info")
        st.metric("Model Accuracy (R²)", f"{model_accuracy:.2%}", "High Performance")
        st.metric("Training Samples", len(df))
        st.metric("Features Used", len(feature_names))
        
        st.markdown("---")
        st.markdown("### 🎯 About")
        st.info("""
        This predictor uses **Gradient Boosting** machine learning 
        to predict final grades (G3) based on 32 different factors including:
        - Previous grades (G1, G2)
        - Study habits
        - Family background
        - Social factors
        """)
        
        st.markdown("---")
        st.markdown("### 📈 Grade Scale")
        st.markdown("""
        - **16-20**: Excellent (A)
        - **14-15**: Very Good (B)
        - **12-13**: Good (C)
        - **10-11**: Satisfactory (D)
        - **0-9**: Needs Improvement
        """)

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Predict Performance", "📊 Model Analytics", "📚 Dataset Explorer"])
    
    with tab1:
        st.markdown('<div class="section-header">📝 Enter Student Information</div>', unsafe_allow_html=True)
        
        # Create form sections
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👤 Basic Information")
            school = st.selectbox("School", ["Gabriel Pereira (GP)", "Mousinho da Silveira (MS)"])
            sex = st.selectbox("Gender", ["Female", "Male"])
            age = st.slider("Age", 15, 22, 17)
            address = st.selectbox("Address Type", ["Rural", "Urban"])
            famsize = st.selectbox("Family Size", ["≤ 3 members", "> 3 members"])
            Pstatus = st.selectbox("Parent Status", ["Apart", "Together"])
            
            st.markdown("#### 🏠 Family Background")
            Medu = st.select_slider("Mother's Education", 
                options=[0, 1, 2, 3, 4],
                format_func=lambda x: ["None", "Primary", "5th-9th", "Secondary", "Higher"][x])
            Fedu = st.select_slider("Father's Education",
                options=[0, 1, 2, 3, 4],
                format_func=lambda x: ["None", "Primary", "5th-9th", "Secondary", "Higher"][x])
            Mjob = st.selectbox("Mother's Job", ["At home", "Health care", "Other", "Services", "Teacher"])
            Fjob = st.selectbox("Father's Job", ["At home", "Health care", "Other", "Services", "Teacher"])
            guardian = st.selectbox("Guardian", ["Father", "Mother", "Other"])
            famrel = st.slider("Family Relationship Quality (1-5)", 1, 5, 4)
            
            st.markdown("#### 📚 Academic Support")
            reason = st.selectbox("Reason for School", ["Close to home", "Course preference", "Other", "Reputation"])
            schoolsup = st.selectbox("Extra Educational Support", ["No", "Yes"])
            famsup = st.selectbox("Family Educational Support", ["No", "Yes"])
            paid = st.selectbox("Extra Paid Classes", ["No", "Yes"])
            activities = st.selectbox("Extra-curricular Activities", ["No", "Yes"])
            nursery = st.selectbox("Attended Nursery", ["No", "Yes"])
            higher = st.selectbox("Want Higher Education?", ["No", "Yes"])
            internet = st.selectbox("Internet at Home?", ["No", "Yes"])
        
        with col2:
            st.markdown("#### 📖 Study Patterns")
            traveltime = st.select_slider("Travel Time to School",
                options=[1, 2, 3, 4],
                format_func=lambda x: ["<15 min", "15-30 min", "30-60 min", ">60 min"][x-1])
            studytime = st.select_slider("Weekly Study Time",
                options=[1, 2, 3, 4],
                format_func=lambda x: ["<2 hrs", "2-5 hrs", "5-10 hrs", ">10 hrs"][x-1])
            failures = st.selectbox("Past Class Failures", [0, 1, 2, 3])
            
            st.markdown("#### 🎯 Previous Performance")
            G1 = st.slider("First Period Grade (G1)", 0, 20, 10)
            G2 = st.slider("Second Period Grade (G2)", 0, 20, 11)
            absences = st.number_input("Number of Absences", 0, 93, 5)
            
            st.markdown("#### 🎭 Lifestyle & Social")
            romantic = st.selectbox("In a Relationship?", ["No", "Yes"])
            freetime = st.slider("Free Time (1-5)", 1, 5, 3)
            goout = st.slider("Going Out with Friends (1-5)", 1, 5, 3)
            Dalc = st.slider("Workday Alcohol Consumption (1-5)", 1, 5, 1)
            Walc = st.slider("Weekend Alcohol Consumption (1-5)", 1, 5, 1)
            health = st.slider("Health Status (1-5)", 1, 5, 4)
        
        # Predict button
        st.markdown("---")
        if st.button("🎯 PREDICT FINAL GRADE (G3)", use_container_width=True):
            # Encode inputs
            input_data = {
                'school': 0 if school == "Gabriel Pereira (GP)" else 1,
                'sex': 0 if sex == "Female" else 1,
                'age': age,
                'address': 0 if address == "Rural" else 1,
                'famsize': 0 if famsize == "≤ 3 members" else 1,
                'Pstatus': 0 if Pstatus == "Apart" else 1,
                'Medu': Medu,
                'Fedu': Fedu,
                'Mjob': ["At home", "Health care", "Other", "Services", "Teacher"].index(Mjob),
                'Fjob': ["At home", "Health care", "Other", "Services", "Teacher"].index(Fjob),
                'reason': ["Close to home", "Course preference", "Other", "Reputation"].index(reason),
                'guardian': ["Father", "Mother", "Other"].index(guardian),
                'traveltime': traveltime,
                'studytime': studytime,
                'failures': failures,
                'schoolsup': 0 if schoolsup == "No" else 1,
                'famsup': 0 if famsup == "No" else 1,
                'paid': 0 if paid == "No" else 1,
                'activities': 0 if activities == "No" else 1,
                'nursery': 0 if nursery == "No" else 1,
                'higher': 0 if higher == "No" else 1,
                'internet': 0 if internet == "No" else 1,
                'romantic': 0 if romantic == "No" else 1,
                'famrel': famrel,
                'freetime': freetime,
                'goout': goout,
                'Dalc': Dalc,
                'Walc': Walc,
                'health': health,
                'absences': absences,
                'G1': G1,
                'G2': G2
            }
            
            # Create DataFrame in correct order
            input_df = pd.DataFrame([input_data])[feature_names]
            
            # Scale and predict
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            
            # Ensure prediction is within bounds
            prediction = max(0, min(20, prediction))
            
            # Display result
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown(f'<p class="prediction-value">{prediction:.1f}</p>', unsafe_allow_html=True)
            st.markdown('<p class="prediction-label">Predicted Final Grade (G3)</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            prev_avg = (G1 + G2) / 2
            change = prediction - prev_avg
            pass_prob = min(95, max(10, 60 + (prediction - 10) * 3)) if prediction >= 10 else max(10, prediction * 5)
            
            if prediction >= 16:
                grade_level = "A - Excellent"
                emoji = "🌟"
            elif prediction >= 14:
                grade_level = "B - Very Good"
                emoji = "⭐"
            elif prediction >= 12:
                grade_level = "C - Good"
                emoji = "✨"
            elif prediction >= 10:
                grade_level = "D - Satisfactory"
                emoji = "👍"
            else:
                grade_level = "Needs Improvement"
                emoji = "📈"
            
            with col1:
                st.metric("Performance Level", f"{emoji} {grade_level}")
            with col2:
                st.metric("Previous Average", f"{prev_avg:.1f}", f"{change:+.1f}")
            with col3:
                st.metric("Expected Change", f"{change:+.1f}", 
                         "Improving" if change > 0 else "Declining")
            with col4:
                st.metric("Pass Probability", f"{pass_prob:.0f}%")
            
            # Recommendations
            st.markdown('<div class="section-header">💡 Personalized Recommendations</div>', unsafe_allow_html=True)
            
            recommendations = []
            
            if failures > 0:
                recommendations.append("⚠️ **Past failures detected.** Consider seeking tutoring or study groups.")
            
            if studytime < 3:
                recommendations.append("📚 **Low study time.** Aim for at least 5-10 hours per week for better results.")
            
            if absences > 10:
                recommendations.append("🚨 **High absences.** Regular attendance is crucial for success.")
            
            if Dalc > 2 or Walc > 3:
                recommendations.append("🍺 **Alcohol consumption** may impact your studies. Consider reducing intake.")
            
            if goout > 3:
                recommendations.append("🎉 **Social activities** are important, but balance them with study time.")
            
            if not (famsup == "Yes"):
                recommendations.append("👨‍👩‍👧 **Family support** can help. Discuss your goals with your family.")
            
            if not (internet == "Yes"):
                recommendations.append("💻 **Internet access** provides valuable resources. Use school/library facilities.")
            
            if health < 3:
                recommendations.append("🏥 **Health concerns.** Ensure proper rest, nutrition, and exercise.")
            
            if prediction >= 14:
                recommendations.append("🎓 **Excellent work!** Keep it up and consider helping peers.")
            elif prediction >= 10:
                recommendations.append("✅ **On track to pass.** Stay consistent and address weak areas.")
            else:
                recommendations.append("🆘 **Immediate action needed.** Seek help from teachers and counselors.")
            
            for rec in recommendations:
                if "🚨" in rec or "⚠️" in rec or "🆘" in rec:
                    st.markdown(f'<div class="warning-box">{rec}</div>', unsafe_allow_html=True)
                elif "🎓" in rec or "✅" in rec:
                    st.markdown(f'<div class="success-box">{rec}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="info-box">{rec}</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="section-header">📊 Model Performance Analytics</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            fig = px.bar(feature_importance, x='Importance', y='Feature', 
                        orientation='h',
                        title='Top 10 Most Important Features',
                        color='Importance',
                        color_continuous_scale='Viridis')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Grade distribution
            fig = px.histogram(df, x='G3', nbins=20,
                             title='Final Grade (G3) Distribution in Dataset',
                             color_discrete_sequence=['#667eea'])
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.markdown('<div class="section-header">🔥 Feature Correlation Heatmap</div>', unsafe_allow_html=True)
        
        # Select numeric columns for correlation
        numeric_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 
                       'failures', 'famrel', 'freetime', 'goout', 'Dalc', 
                       'Walc', 'health', 'absences', 'G1', 'G2', 'G3']
        
        corr_data = df[numeric_cols].corr()
        
        fig = px.imshow(corr_data, 
                       color_continuous_scale='RdBu_r',
                       title='Correlation Matrix of Key Features',
                       aspect='auto')
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Model metrics
        st.markdown('<div class="section-header">📈 Model Performance Metrics</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("R² Score", f"{model_accuracy:.4f}", "High Accuracy")
        with col2:
            st.metric("Model Type", "Gradient Boosting")
        with col3:
            st.metric("Features Used", len(feature_names))
    
    with tab3:
        st.markdown('<div class="section-header">📚 Dataset Explorer</div>', unsafe_allow_html=True)
        
        st.markdown("### 📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Students", len(df))
        with col2:
            st.metric("Average G3", f"{df['G3'].mean():.2f}")
        with col3:
            st.metric("Pass Rate", f"{(df['G3'] >= 10).sum() / len(df) * 100:.1f}%")
        with col4:
            st.metric("Features", df.shape[1])
        
        st.markdown("### 🔍 Sample Data")
        st.dataframe(df.head(20), use_container_width=True)
        
        st.markdown("### 📉 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Grade analysis
        st.markdown("### 📊 Grade Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # G1 vs G3
            fig = px.scatter(df, x='G1', y='G3', 
                           title='First Period Grade (G1) vs Final Grade (G3)',
                           trendline='ols',
                           color='G3',
                           color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # G2 vs G3
            fig = px.scatter(df, x='G2', y='G3',
                           title='Second Period Grade (G2) vs Final Grade (G3)',
                           trendline='ols',
                           color='G3',
                           color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)

else:
    st.error("⚠️ Failed to load model. Please ensure 'student_data.csv' is in the same directory as this script.")
    st.info("""
    ### How to fix:
    1. Make sure `student_data.csv` is in the same folder as `app.py`
    2. The CSV should contain all required columns
    3. Restart the Streamlit app
    """)