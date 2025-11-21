# 🧠 ASD Comorbidity & SDOH Interactive Dashboard

## Quantifying the Influence of Socioeconomic Factors on Comorbidities in Children with Autism Using Explainable AI

**Author:** Rohan Dhameja  
**Institution:** Bellarmine College Preparatory High School, San Jose, CA

---

## 📋 Project Overview

This interactive dashboard visualizes research findings on how Social Determinants of Health (SDOH) influence comorbidities (ADHD, anxiety, depression, and epilepsy) in children with Autism Spectrum Disorder (ASD).

### Key Features:

- **📊 Overview Dashboard**: Dataset statistics and comorbidity prevalence
- **🤖 Model Performance**: Comparison of Logistic Regression, Random Forest, and XGBoost
- **🎯 SHAP Analysis**: Interactive explainable AI visualizations showing feature importance
- **📈 Feature Explorer**: Deep dive into individual SDOH factors
- **🗂️ Data Explorer**: Interactive data filtering and exploration

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd /Users/asd-comorbidity-analysis
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser:**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in your terminal

---

## 📊 Dashboard Pages

### 1. 📊 Overview
- Total participants and dataset statistics
- Comorbidity prevalence visualization
- Distribution of SDOH variables
- Categorical variable breakdowns

### 2. 🤖 Model Performance
- Performance metrics for all three models
- Interactive comparison charts
- Best model identification
- Detailed evaluation metrics:
  - Precision
  - Recall
  - F1-Score
  - Hamming Loss

### 3. 🎯 SHAP Analysis
- Feature importance for each comorbidity
- Interactive SHAP value visualization
- Top contributing features
- Heatmap of feature importance across all comorbidities
- Model selection (Random Forest/XGBoost)

### 4. 📈 Feature Explorer
- Individual feature analysis
- Relationship between SDOH factors and comorbidities
- Statistical summaries
- Multi-feature comparison
- Correlation analysis

### 5. 🗂️ Data Explorer
- Interactive data filtering
- Dynamic prevalence calculations
- Exportable filtered datasets
- Summary statistics
- Custom column selection

---

## 📁 Project Structure

```
asd-comorbidity-analysis/
├── app.py                             # Main Streamlit dashboard
├── asd_comorbidity_analysis.ipynb    # Jupyter notebook with analysis
├── Autism_SDOH_Comorbidity.csv       # Dataset (2,568 ASD-positive children)
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## 📈 Dataset Information

**Source:** 2020-2021 National Survey of Children's Health (NSCH)

**Participants:**
- Total: 79,182 children
- ASD-positive: 2,568 children

**Features:**
- 11 SDOH variables (age, gender, SES, parental education, family income, etc.)
- 4 comorbidity outcomes (ADHD, anxiety, depression, epilepsy)

**SDOH Variables:**
1. Age
2. Gender
3. Race/Ethnicity
4. Socioeconomic Status (SES)
5. Neighborhood Resources
6. Housing Stability
7. Insurance Status
8. Family Income (normalized)
9. Parental Education
10. Neighborhood Characteristics
11. Healthcare Access

---

## 🔬 Methodology

### Machine Learning Models:
1. **Logistic Regression** (max_iter=1000)
2. **Random Forest** (n_estimators=200)
3. **XGBoost** (n_estimators=300, learning_rate=0.01)

### Multi-Output Classification:
All models use `MultiOutputClassifier` to simultaneously predict all four comorbidities.

### Explainable AI:
SHAP (SHapley Additive exPlanations) values are computed to identify the most influential SDOH factors.

---

## 📊 Key Findings

From the research analysis:

✅ **Logistic Regression** achieved the highest recall (1.000) and competitive F1-score (0.897)

✅ **Parental Education** and **Family Income** identified as the most influential SDOH factors

✅ Higher socioeconomic resources consistently correlate with lower comorbidity risk

✅ Multi-output modeling captures interdependencies between comorbidities

---

## 🎯 Use Cases

### For Researchers:
- Explore feature importance across different comorbidities
- Compare model performance
- Generate publication-ready visualizations

### For Clinicians:
- Understand SDOH risk factors
- Identify at-risk populations
- Guide intervention strategies

### For Policymakers:
- Evidence-based resource allocation
- Target socioeconomically disadvantaged families
- Inform public health strategies

---

## 🌐 Deployment Options

### Local Deployment (Current):
```bash
streamlit run app.py
```

### Cloud Deployment:

#### Option 1: Streamlit Cloud (Free)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click

#### Option 2: Heroku
```bash
heroku create
git push heroku main
```

#### Option 3: AWS/GCP/Azure
- Deploy as containerized application
- Use managed services (AWS Elastic Beanstalk, GCP App Engine, etc.)

---

## 🛠️ Customization

### Adding New Features:
Edit `app.py` and add new pages or visualizations in the respective function sections.

### Changing Models:
Modify the `train_models()` function to add or update ML algorithms.

### Updating Data:
Replace `Autism_SDOH_Comorbidity.csv` with your own dataset (maintain same column structure).

---

## 📝 Citation

If you use this dashboard or research in your work, please cite:

```
Dhameja, R. (2025). Quantifying the Influence of Socioeconomic Factors on 
Comorbidities in Children with Autism Using Explainable AI. 
Bellarmine College Preparatory High School, San Jose, CA.
```

---

## 📧 Contact

**Rohan Dhameja**  
Bellarmine College Preparatory High School  
San Jose, CA

---

## 🙏 Acknowledgments

- National Survey of Children's Health (NSCH) 2020-2021 dataset
- Data Resource Center for Child and Adolescent Health
- Stanford Conference reviewers and mentors

---

## 📄 License

This project is for educational and research purposes.

---

## 🐛 Troubleshooting

### Issue: Package installation fails
**Solution:** Upgrade pip first:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Issue: Port already in use
**Solution:** Specify a different port:
```bash
streamlit run app.py --server.port 8502
```

### Issue: SHAP visualizations not showing
**Solution:** Make sure you selected Random Forest or XGBoost model in the SHAP Analysis page.

### Issue: Large dataset loading slowly
**Solution:** The app uses caching (`@st.cache_data`) to speed up subsequent loads.

---

**Built with ❤️ using Streamlit, scikit-learn, SHAP, and Plotly**


