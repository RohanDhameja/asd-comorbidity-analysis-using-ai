"""
Interactive Dashboard for Autism Spectrum Disorder (ASD) Comorbidity Analysis
Quantifying the Influence of Socioeconomic Factors on Comorbidities in Children with Autism

Author: Rohan Dhameja
Institution: Bellarmine College Preparatory High School
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss
from xgboost import XGBClassifier

# Page configuration
st.set_page_config(
    page_title="ASD Comorbidity & SDOH Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Global Styles */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    
    /* Home Page Styles */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .hero-subtitle {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .profile-card {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .achievement-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        display: inline-block;
        margin: 0.5rem;
        font-weight: bold;
    }
    .research-highlight {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        margin: 1rem 0;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        transition: transform 0.3s;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    </style>
""", unsafe_allow_html=True)

# Caching functions for performance
@st.cache_data
def load_data():
    """Load the autism SDOH comorbidity dataset"""
    data = pd.read_csv("Autism_SDOH_Comorbidity.csv")
    return data

@st.cache_data
def preprocess_data(data):
    """Preprocess data: encoding and scaling"""
    target_cols = ["ADHD", "anxiety", "depression", "epilepsy"]
    X = data.drop(columns=["Child_ID"] + target_cols)
    y = data[target_cols]
    
    # Store original feature names
    feature_names = X.columns.tolist()
    
    # Encode categorical variables
    cat_cols = X.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Scale continuous variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
    
    return X_scaled_df, y, feature_names, label_encoders, scaler

@st.cache_resource
def train_models(X_train, y_train):
    """Train all three models"""
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.01, random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        multi_model = MultiOutputClassifier(model)
        multi_model.fit(X_train, y_train)
        trained_models[name] = multi_model
    
    return trained_models

@st.cache_data
def evaluate_models(_models, X_test, y_test):
    """Evaluate all models and return metrics"""
    results = []
    target_cols = ["ADHD", "anxiety", "depression", "epilepsy"]
    
    for name, model in _models.items():
        y_pred = model.predict(X_test)
        
        results.append({
            "Model": name,
            "Precision": precision_score(y_test, y_pred, average='micro'),
            "Recall": recall_score(y_test, y_pred, average='micro'),
            "F1-score": f1_score(y_test, y_pred, average='micro'),
            "Hamming Loss": hamming_loss(y_test, y_pred)
        })
    
    return pd.DataFrame(results)

@st.cache_data
def compute_shap_values(_model, X_test, feature_names):
    """Compute SHAP values for each comorbidity"""
    target_cols = ["ADHD", "anxiety", "depression", "epilepsy"]
    shap_values_dict = {}
    
    # Convert X_test to numpy array if it's a DataFrame
    if isinstance(X_test, pd.DataFrame):
        X_test_array = X_test.values
    else:
        X_test_array = X_test
    
    for i, comorbidity in enumerate(target_cols):
        estimator = _model.estimators_[i]
        explainer = shap.TreeExplainer(estimator)
        shap_vals = explainer.shap_values(X_test_array)
        
        # For binary classification, SHAP returns values for both classes
        # We want class 1 (positive class - has comorbidity)
        if isinstance(shap_vals, list):
            # List of arrays [class_0_values, class_1_values]
            if len(shap_vals) == 2:
                shap_values_dict[comorbidity] = shap_vals[1]
            else:
                shap_values_dict[comorbidity] = shap_vals[0]
        elif isinstance(shap_vals, np.ndarray):
            # Check if it's already the right shape (samples, features)
            if len(shap_vals.shape) == 2 and shap_vals.shape[1] == len(feature_names):
                shap_values_dict[comorbidity] = shap_vals
            # If shape is (samples, features*2), take second half for positive class
            elif len(shap_vals.shape) == 2 and shap_vals.shape[1] == len(feature_names) * 2:
                shap_values_dict[comorbidity] = shap_vals[:, len(feature_names):]
            else:
                # Default: use as is
                shap_values_dict[comorbidity] = shap_vals
        else:
            shap_values_dict[comorbidity] = shap_vals
    
    return shap_values_dict

# ==================== MAIN APP ====================

def main():
    # Title
    st.markdown('<div class="main-header">🧠 ASD Comorbidity & SDOH Interactive Dashboard</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <b>Research:</b> Quantifying the Influence of Socioeconomic Factors on Comorbidities in Children with Autism Using Explainable AI<br>
    <b>Author:</b> Rohan Dhameja | <b>Institution:</b> Bellarmine College Preparatory High School
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🧠 ASD Comorbidity Analysis")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "📍 Navigation:",
        ["🏠 Home", "📊 Overview", "🤖 Model Performance", "🎯 SHAP Analysis", "📈 Feature Explorer", "🗂️ Data Explorer"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        ### 👨‍🔬 Researcher
        **Rohan Dhameja**  
        Bellarmine College Prep
        
        ### 📊 Dataset
        - 79,182 total participants
        - 2,568 ASD-positive children
        - 11 predictor variables
        - 4 comorbidities analyzed
        
        ### 🎯 Conference
        Stanford Research Conference
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Tip:** Start with Home to learn about the research!")
    
    # Show home page without loading data
    if page == "🏠 Home":
        show_home()
    else:
        # Load and preprocess data only for analysis pages
        with st.spinner("Loading data..."):
            data = load_data()
            X, y, feature_names, label_encoders, scaler = preprocess_data(data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Train models
        with st.spinner("Training models..."):
            models = train_models(X_train, y_train)
        
        # Route to different pages
        if page == "📊 Overview":
            show_overview(data, y)
        elif page == "🤖 Model Performance":
            show_model_performance(models, X_test, y_test)
        elif page == "🎯 SHAP Analysis":
            show_shap_analysis(models, X_test, feature_names)
        elif page == "📈 Feature Explorer":
            show_feature_explorer(data, models, X_test, feature_names)
        elif page == "🗂️ Data Explorer":
            show_data_explorer(data)

# ==================== HOME PAGE ====================

def show_home():
    """Home page with researcher profile and introduction"""
    
    # Hero Section
    st.markdown("""
        <div class="hero-section">
            <div class="hero-title">🧠 ASD Comorbidity Analysis</div>
            <div class="hero-subtitle">Quantifying the Influence of Socioeconomic Factors Using Explainable AI</div>
            <p style="font-size: 1.2rem; margin-top: 1rem;">Interactive ML Dashboard for Understanding Autism Comorbidities</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Profile Section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
            <div class="profile-card">
                <h2 style="color: #667eea; margin-bottom: 1rem;">👨‍🔬 Researcher Profile</h2>
                <h3 style="color: #333;">Rohan Dhameja</h3>
                <p style="color: #666; margin-bottom: 1rem;">
                    <strong>Institution:</strong><br>
                    Bellarmine College Preparatory<br>
                    High School<br>
                    San Jose, CA
                </p>
                <p style="color: #666;">
                    <strong>Contact:</strong><br>
                    rohan.dhameja27@bcp.org
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Achievements
        st.markdown("""
            <div style="margin-top: 1rem;">
                <div class="achievement-badge">🏆 Stanford Conference</div>
                <div class="achievement-badge">🎓 ML Researcher</div>
                <div class="achievement-badge">🧬 Healthcare AI</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="research-highlight">
                <h2 style="margin-bottom: 1rem;">📊 Research Overview</h2>
                <p style="font-size: 1.1rem; line-height: 1.8;">
                Autism Spectrum Disorder (ASD) is commonly accompanied by comorbid conditions including 
                <strong>ADHD, anxiety, depression, and epilepsy</strong>, which intensify functional impairments 
                and complicate clinical management.
                </p>
                <p style="font-size: 1.1rem; line-height: 1.8; margin-top: 1rem;">
                This research explores how <strong>Social Determinants of Health (SDOH)</strong>—such as 
                parental education, household income, and healthcare access—influence these comorbidity risks 
                among children with autism.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Key Findings
    st.markdown("## 🔬 Key Research Components")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="feature-card">
                <h3>📈 Dataset</h3>
                <ul style="text-align: left; margin-left: 1rem;">
                    <li><strong>79,182</strong> participants</li>
                    <li><strong>2,568</strong> ASD-positive children</li>
                    <li>National Survey of Children's Health (2020-2021)</li>
                    <li>11 socioeconomic predictors</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card">
                <h3>🤖 ML Models</h3>
                <ul style="text-align: left; margin-left: 1rem;">
                    <li>Logistic Regression</li>
                    <li>Random Forest</li>
                    <li>XGBoost</li>
                    <li>Multi-output classification</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="feature-card">
                <h3>🎯 Key Findings</h3>
                <ul style="text-align: left; margin-left: 1rem;">
                    <li>Parental education is #1 factor</li>
                    <li>Family income is critical</li>
                    <li>100% recall achieved</li>
                    <li>SHAP for interpretability</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # Dashboard Features
    st.markdown("## 🎛️ Dashboard Features")
    
    features = [
        ("📊 Overview", "Explore dataset statistics and comorbidity prevalence"),
        ("🤖 Model Performance", "Compare ML models with precision, recall, and F1-scores"),
        ("🎯 SHAP Analysis", "Understand feature importance with explainable AI"),
        ("📈 Feature Explorer", "Analyze relationships between socioeconomic factors"),
        ("🗂️ Data Explorer", "Filter and download custom dataset views")
    ]
    
    for feature, description in features:
        st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; border-left: 4px solid #667eea;">
                <strong style="color: #667eea; font-size: 1.2rem;">{feature}</strong>
                <p style="color: #666; margin: 0.5rem 0 0 0;">{description}</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Research Impact
    st.markdown("## 💡 Research Impact")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Clinical Applications:**
        - Early intervention strategies
        - Risk stratification tools
        - Evidence-based resource allocation
        - Targeted support for at-risk families
        """)
    
    with col2:
        st.success("""
        **Policy Implications:**
        - Address socioeconomic disparities
        - Improve healthcare accessibility
        - Support parental education programs
        - Inform public health strategies
        """)
    
    # Call to Action
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 1rem; color: white;">
            <h2 style="margin-bottom: 1rem;">🚀 Ready to Explore?</h2>
            <p style="font-size: 1.2rem;">Use the sidebar to navigate through different analyses and discover insights!</p>
        </div>
    """, unsafe_allow_html=True)

# ==================== PAGE 1: OVERVIEW ====================

def show_overview(data, y):
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Participants", f"{len(data):,}")
    with col2:
        st.metric("SDOH Features", "11")
    with col3:
        st.metric("Comorbidities Analyzed", "4")
    with col4:
        st.metric("Average Comorbidities", f"{y.sum(axis=1).mean():.2f}")
    
    st.markdown("---")
    
    # Comorbidity prevalence
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Comorbidity Prevalence in ASD-Positive Children")
        
        comorbidity_rates = y.mean() * 100
        
        fig = go.Figure(data=[
            go.Bar(
                x=comorbidity_rates.values,
                y=[c.upper() for c in comorbidity_rates.index],
                orientation='h',
                marker=dict(
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'],
                    line=dict(color='rgba(0,0,0,0.3)', width=1)
                ),
                text=[f"{v:.1f}%" for v in comorbidity_rates.values],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Prevalence Rate (%)",
            xaxis_title="Percentage (%)",
            yaxis_title="Comorbidity",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Key Statistics")
        
        for comorbidity in y.columns:
            count = y[comorbidity].sum()
            rate = (count / len(y)) * 100
            st.markdown(f"""
            <div class="metric-card">
                <b>{comorbidity.upper()}</b><br>
                {count:,} cases ({rate:.1f}%)
            </div>
            """, unsafe_allow_html=True)
            st.write("")
    
    st.markdown("---")
    
    # SDOH Distribution
    st.subheader("Social Determinants of Health Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Family income distribution
        fig = px.histogram(
            data, 
            x="family_income",
            nbins=30,
            title="Family Income Distribution (Normalized)",
            labels={"family_income": "Family Income", "count": "Frequency"},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Age distribution
        fig = px.histogram(
            data,
            x="age",
            nbins=15,
            title="Age Distribution of Children",
            labels={"age": "Age (years)", "count": "Frequency"},
            color_discrete_sequence=['#ff7f0e']
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Categorical SDOH
    st.subheader("Categorical SDOH Variables")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = px.pie(
            data,
            names="SES",
            title="Socioeconomic Status",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(
            data,
            names="parental_education",
            title="Parental Education",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = px.pie(
            data,
            names="insurance_status",
            title="Insurance Status",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE 2: MODEL PERFORMANCE ====================

def show_model_performance(models, X_test, y_test):
    st.header("🤖 Model Performance Comparison")
    
    # Evaluate models
    results_df = evaluate_models(models, X_test, y_test)
    
    # Display metrics table
    st.subheader("Performance Metrics")
    
    # Style the dataframe
    styled_df = results_df.style.background_gradient(
        subset=['Precision', 'Recall', 'F1-score'],
        cmap='RdYlGn',
        vmin=0.7,
        vmax=1.0
    ).background_gradient(
        subset=['Hamming Loss'],
        cmap='RdYlGn_r',
        vmin=0.1,
        vmax=0.3
    ).format({
        'Precision': '{:.5f}',
        'Recall': '{:.5f}',
        'F1-score': '{:.5f}',
        'Hamming Loss': '{:.5f}'
    })
    
    st.dataframe(styled_df, use_container_width=True)
    
    st.markdown("---")
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Metrics Comparison")
        
        fig = go.Figure()
        
        metrics = ['Precision', 'Recall', 'F1-score']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for metric, color in zip(metrics, colors):
            fig.add_trace(go.Bar(
                name=metric,
                x=results_df['Model'],
                y=results_df[metric],
                marker_color=color,
                text=[f"{v:.3f}" for v in results_df[metric]],
                textposition='auto',
            ))
        
        fig.update_layout(
            barmode='group',
            title="Multi-Output Classification Performance",
            xaxis_title="Model",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1.1]),
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Hamming Loss Comparison")
        
        fig = go.Figure(data=[
            go.Bar(
                x=results_df['Model'],
                y=results_df['Hamming Loss'],
                marker=dict(
                    color=results_df['Hamming Loss'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Loss")
                ),
                text=[f"{v:.5f}" for v in results_df['Hamming Loss']],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Hamming Loss (Lower is Better)",
            xaxis_title="Model",
            yaxis_title="Hamming Loss",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Best model
    best_model = results_df.loc[results_df['F1-score'].idxmax()]
    
    st.markdown("---")
    st.subheader("🏆 Best Performing Model")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #1f77b4;">{best_model['Model']}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Precision", f"{best_model['Precision']:.5f}")
    
    with col3:
        st.metric("Recall", f"{best_model['Recall']:.5f}")
    
    with col4:
        st.metric("F1-Score", f"{best_model['F1-score']:.5f}")
    
    st.info("""
    **Interpretation:**
    - **Precision**: Proportion of predicted comorbidities that are actually correct
    - **Recall**: Proportion of actual comorbidities that were correctly identified
    - **F1-Score**: Harmonic mean of precision and recall
    - **Hamming Loss**: Fraction of incorrectly predicted labels
    """)

# ==================== PAGE 3: SHAP ANALYSIS ====================

def show_shap_analysis(models, X_test, feature_names):
    st.header("🎯 SHAP Analysis: Feature Importance")
    
    st.info("""
    **SHAP (SHapley Additive exPlanations)** values quantify the contribution of each feature to the model's predictions.
    Higher absolute SHAP values indicate more influential features.
    """)
    
    # Model selection
    model_choice = st.selectbox(
        "Select Model for SHAP Analysis:",
        ["Random Forest", "Logistic Regression", "XGBoost"]
    )
    
    # Only compute SHAP for tree-based models
    if model_choice in ["Random Forest", "XGBoost"]:
        with st.spinner(f"Computing SHAP values for {model_choice}..."):
            shap_values_dict = compute_shap_values(models[model_choice], X_test, feature_names)
        
        # Comorbidity selection
        comorbidity = st.selectbox(
            "Select Comorbidity:",
            ["ADHD", "anxiety", "depression", "epilepsy"],
            format_func=lambda x: x.upper()
        )
        
        st.markdown("---")
        
        # Feature importance for selected comorbidity
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Feature Importance for {comorbidity.upper()}")
            
            shap_vals = shap_values_dict[comorbidity]
            
            # Debug info (optional - can be removed later)
            # st.caption(f"SHAP values shape: {shap_vals.shape}")
            
            # Calculate mean absolute SHAP values
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            
            # Ensure mean_abs_shap is 1D
            if len(mean_abs_shap.shape) > 1:
                mean_abs_shap = mean_abs_shap.flatten()
            
            # Handle case where SHAP values are doubled (e.g., 22 instead of 11)
            if len(mean_abs_shap) == len(feature_names) * 2:
                # Take the second half (positive class)
                mean_abs_shap = mean_abs_shap[len(feature_names):]
                st.info("Automatically adjusted SHAP values for binary classification.")
            
            # Ensure lengths match
            if len(feature_names) != len(mean_abs_shap):
                st.error(f"Feature mismatch: {len(feature_names)} features but {len(mean_abs_shap)} SHAP values")
                st.error(f"SHAP array shape: {shap_vals.shape}, Expected features: {len(feature_names)}")
                st.info("Try selecting a different model or comorbidity.")
                return
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': mean_abs_shap
            }).sort_values('Importance', ascending=False)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=importance_df['Importance'],
                    y=importance_df['Feature'],
                    orientation='h',
                    marker=dict(
                        color=importance_df['Importance'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Importance")
                    ),
                    text=[f"{v:.4f}" for v in importance_df['Importance']],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                xaxis_title="Mean |SHAP Value|",
                yaxis_title="Feature",
                height=600,
                yaxis=dict(autorange="reversed")
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Top 5 Features")
            
            top_5 = importance_df.head(5)
            
            for idx, (_, row) in enumerate(top_5.iterrows(), 1):
                st.markdown(f"""
                <div class="metric-card">
                    <b>#{idx}: {row['Feature']}</b><br>
                    Impact: {row['Importance']:.4f}
                </div>
                """, unsafe_allow_html=True)
                st.write("")
            
            st.markdown("---")
            st.subheader("Insights")
            
            top_feature = importance_df.iloc[0]['Feature']
            st.success(f"**{top_feature}** is the most influential SDOH factor for predicting {comorbidity.upper()} in this model.")
        
        st.markdown("---")
        
        # Comparative analysis across all comorbidities
        st.subheader("Feature Importance Across All Comorbidities")
        
        # Collect importance for all comorbidities
        importance_matrix = []
        target_cols = ["ADHD", "anxiety", "depression", "epilepsy"]
        
        for comorb in target_cols:
            shap_vals = shap_values_dict[comorb]
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            # Ensure 1D
            if len(mean_abs_shap.shape) > 1:
                mean_abs_shap = mean_abs_shap.flatten()
            # Handle doubled SHAP values
            if len(mean_abs_shap) == len(feature_names) * 2:
                mean_abs_shap = mean_abs_shap[len(feature_names):]
            importance_matrix.append(mean_abs_shap)
        
        importance_df_all = pd.DataFrame(
            importance_matrix,
            columns=feature_names,
            index=[c.upper() for c in target_cols]
        ).T
        
        # Sort by average importance
        importance_df_all['Average'] = importance_df_all.mean(axis=1)
        importance_df_sorted = importance_df_all.sort_values('Average', ascending=False).head(10)
        importance_df_display = importance_df_sorted.drop('Average', axis=1)
        
        # Heatmap
        fig = px.imshow(
            importance_df_display.T,
            labels=dict(x="Feature", y="Comorbidity", color="SHAP Value"),
            x=importance_df_display.index,
            y=importance_df_display.columns,
            color_continuous_scale='YlOrRd',
            aspect="auto"
        )
        
        fig.update_layout(
            title="Top 10 Features by Average Importance",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display table
        st.dataframe(
            importance_df_display.style.background_gradient(cmap='YlOrRd').format('{:.4f}'),
            use_container_width=True
        )
        
    else:
        st.warning("⚠️ SHAP analysis is currently only available for Random Forest and XGBoost models.")
        st.info("Please select Random Forest or XGBoost from the dropdown above.")

# ==================== PAGE 4: FEATURE EXPLORER ====================

def show_feature_explorer(data, models, X_test, feature_names):
    st.header("📈 Interactive Feature Explorer")
    
    st.markdown("""
    Explore how individual SDOH features relate to comorbidity outcomes.
    """)
    
    # Feature selection
    feature = st.selectbox(
        "Select a Feature to Explore:",
        feature_names
    )
    
    comorbidity = st.selectbox(
        "Select Comorbidity:",
        ["ADHD", "anxiety", "depression", "epilepsy"],
        format_func=lambda x: x.upper(),
        key="feature_explorer_comorbidity"
    )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{feature} vs {comorbidity.upper()}")
        
        # Create visualization based on feature type
        if data[feature].dtype == 'object':
            # Categorical feature
            plot_data = data.groupby([feature, comorbidity]).size().reset_index(name='count')
            
            fig = px.bar(
                plot_data,
                x=feature,
                y='count',
                color=comorbidity,
                barmode='group',
                title=f"Distribution of {comorbidity.upper()} by {feature}",
                labels={comorbidity: f"Has {comorbidity.upper()}", 'count': 'Count'},
                color_discrete_map={0: '#90EE90', 1: '#FF6B6B'}
            )
            
        else:
            # Continuous feature
            fig = px.box(
                data,
                x=comorbidity,
                y=feature,
                color=comorbidity,
                title=f"{feature} Distribution by {comorbidity.upper()} Status",
                labels={comorbidity: f"Has {comorbidity.upper()}"},
                color_discrete_map={0: '#90EE90', 1: '#FF6B6B'}
            )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Statistical Summary")
        
        if data[feature].dtype == 'object':
            # For categorical
            summary = data.groupby(feature)[comorbidity].agg(['count', 'sum', 'mean'])
            summary.columns = ['Total', 'With Comorbidity', 'Prevalence Rate']
            summary['Prevalence Rate'] = (summary['Prevalence Rate'] * 100).round(2)
            
            st.dataframe(
                summary.style.background_gradient(subset=['Prevalence Rate'], cmap='RdYlGn_r'),
                use_container_width=True
            )
            
        else:
            # For continuous
            summary = data.groupby(comorbidity)[feature].describe()
            st.dataframe(summary.T.style.format('{:.4f}'), use_container_width=True)
        
        st.markdown("---")
        
        # Correlation info
        if data[feature].dtype != 'object':
            corr = data[[feature, comorbidity]].corr().iloc[0, 1]
            
            st.metric(
                label="Correlation with Comorbidity",
                value=f"{corr:.4f}",
                delta="Positive" if corr > 0 else "Negative"
            )
    
    st.markdown("---")
    
    # Multi-feature comparison
    st.subheader("Multi-Feature Comparison")
    
    selected_features = st.multiselect(
        "Select features to compare:",
        [f for f in feature_names if data[f].dtype != 'object'],
        default=[feature_names[i] for i in [1, 8] if data[feature_names[i]].dtype != 'object'][:2]
    )
    
    if len(selected_features) >= 2:
        fig = px.scatter(
            data,
            x=selected_features[0],
            y=selected_features[1],
            color=comorbidity,
            title=f"Relationship between {selected_features[0]} and {selected_features[1]}",
            labels={comorbidity: f"Has {comorbidity.upper()}"},
            color_discrete_map={0: '#90EE90', 1: '#FF6B6B'},
            opacity=0.6
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE 5: DATA EXPLORER ====================

def show_data_explorer(data):
    st.header("🗂️ Data Explorer")
    
    st.markdown("""
    Explore and filter the dataset interactively.
    """)
    
    # Filters
    st.subheader("Apply Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age_range = st.slider(
            "Age Range:",
            int(data['age'].min()),
            int(data['age'].max()),
            (int(data['age'].min()), int(data['age'].max()))
        )
    
    with col2:
        ses_filter = st.multiselect(
            "Socioeconomic Status:",
            options=data['SES'].unique(),
            default=data['SES'].unique()
        )
    
    with col3:
        gender_filter = st.multiselect(
            "Gender:",
            options=data['gender'].unique(),
            default=data['gender'].unique()
        )
    
    # Apply filters
    filtered_data = data[
        (data['age'] >= age_range[0]) &
        (data['age'] <= age_range[1]) &
        (data['SES'].isin(ses_filter)) &
        (data['gender'].isin(gender_filter))
    ]
    
    st.markdown("---")
    
    # Display stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Filtered Participants", f"{len(filtered_data):,}")
    with col2:
        st.metric("% of Total", f"{(len(filtered_data)/len(data)*100):.1f}%")
    with col3:
        st.metric("Avg Age", f"{filtered_data['age'].mean():.1f}")
    with col4:
        st.metric("Avg Family Income", f"{filtered_data['family_income'].mean():.3f}")
    
    st.markdown("---")
    
    # Comorbidity summary for filtered data
    st.subheader("Comorbidity Prevalence in Filtered Data")
    
    comorbidities = ["ADHD", "anxiety", "depression", "epilepsy"]
    filtered_prevalence = filtered_data[comorbidities].mean() * 100
    
    fig = go.Figure(data=[
        go.Bar(
            x=[c.upper() for c in comorbidities],
            y=filtered_prevalence.values,
            marker=dict(
                color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'],
            ),
            text=[f"{v:.1f}%" for v in filtered_prevalence.values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Comorbidity Prevalence (%)",
        xaxis_title="Comorbidity",
        yaxis_title="Prevalence (%)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Data table
    st.subheader("Filtered Data Table")
    
    # Column selection
    show_columns = st.multiselect(
        "Select columns to display:",
        options=list(data.columns),
        default=['Child_ID', 'age', 'gender', 'SES', 'parental_education', 'family_income', 
                 'ADHD', 'anxiety', 'depression', 'epilepsy']
    )
    
    # Display dataframe
    st.dataframe(
        filtered_data[show_columns].head(100),
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = filtered_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_autism_sdoh_data.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # Summary statistics
    st.subheader("Summary Statistics")
    
    tab1, tab2 = st.tabs(["Numerical Features", "Categorical Features"])
    
    with tab1:
        st.dataframe(
            filtered_data.describe().T.style.format('{:.3f}'),
            use_container_width=True
        )
    
    with tab2:
        cat_cols = filtered_data.select_dtypes(include=['object']).columns
        for col in cat_cols:
            with st.expander(f"📊 {col}"):
                value_counts = filtered_data[col].value_counts()
                st.dataframe(
                    pd.DataFrame({
                        'Value': value_counts.index,
                        'Count': value_counts.values,
                        'Percentage': (value_counts.values / len(filtered_data) * 100).round(2)
                    }),
                    use_container_width=True
                )

# ==================== RUN APP ====================

if __name__ == "__main__":
    main()

