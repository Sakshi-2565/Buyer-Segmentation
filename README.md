# Buyer-Segmentation

# Premium Machine Learning Real Estate Buyer Segmentation & Market Intelligence Platform

Welcome to this end-to-end, high-end business intelligence platform designed to resemble professional Tableau and PowerBI analytics suites. This platform utilizes Unsupervised Machine Learning (K-Means & Hierarchical Agglomerative Clustering) to segment real estate buyer portfolios into 4 actionable investor personas, accompanied by **30 advanced interactive visualizations**.

---

## 🏢 Platform Overview
Understanding buyer demographics, behavioral indicators, and sentiment preferences is vital to optimize corporate real estate decisions. This platform provides real estate developers, investors, and marketers with:
- **Executive Decision KPIs**: High-end metric cards calculating Total Capital Spend, Mean Spend, Survey Satisfactions, and mortgage loan leverage rates.
- **Advanced Sidebar Filter Board**: Slice all dashboard views dynamically by Country, Region, Client Type, Gender, Purpose, Loan applied, and Satisfaction range.
- **Interactive Visual Lab**: A playground loaded with **30 comprehensive charts** (including 3D PCA, Sankey Flows, Radar, Sunburst, Treemap, and parallel coordinate graphs).
- **AI Automated Insights Engine**: Dynamic natural language audits mapping cohorts to corporate investment recommendations, hidden trends, and behavioral anomalies.

---

## 📂 Project Structure
```
c:\Users\Lenovo\OneDrive\Desktop\Segment\
│
├── data/
│   └── buyers_dataset.csv             # Cleaned and enriched buyers dataset used for ML
│
├── notebooks/
│   └── buyer_segmentation.ipynb       # Explanatory Jupyter Notebook with 20+ diagnostics
│
├── assets/
│   └── css_styles.css                 # Premium custom modern dark theme stylesheet (Glassmorphism)
│
├── src/
│   ├── preprocessing.py               # Preprocessing pipelines (cleaning, scaling, encoding)
│   ├── clustering.py                  # Clustering models (K-Means, Agglomerative, linkage matrix)
│   ├── visualization.py               # Visual analytics library hosting all 30 plotting scripts
│   ├── insights.py                    # Automated Business Insights & Recommendations Engine
│   └── dashboard_utils.py             # Shared KPI calculators & CSS injectors
│
├── app.py                             # Premium Streamlit multi-page BI dashboard
├── requirements.txt                   # Platform library dependencies
└── README.md                          # Platform documentation (this file!)
```

---

## ⚙️ Installation & Run Commands
Setting up and running the dashboard is extremely simple:

### 1. Install Dependencies
Open your terminal in the project root directory and install requirements:
```bash
pip install -r requirements.txt
```

### 2. Generate/Process Consolidated Dataset
Verify that `data/buyers_dataset.csv` exists or regenerate it by executing:
```bash
python -c "import sys; sys.path.append('src'); from utils import load_and_preprocess_raw_data; load_and_preprocess_raw_data()"
```

### 3. Launch the Streamlit Console
To experience the premium dark-themed BI dashboard, launch Streamlit:
```bash
streamlit run app.py
```
*(If streamlit isn't globally mapped, use: `python -m streamlit run app.py`)*

---

## 📈 Multi-Page Intelligence Consoles

The application features **6 professional pages**:

### 1. Executive Overview
- Displays glowing high-value metrics (Total Buyers, Avg Age, Avg Satisfaction, Avg Portfolio Spend, Loan rate).
- Auto-generated summary of top dynamic findings.
- Global treemap showing nesting relationships (Country → Region → Client Type).

### 2. Buyer Segmentation Analytics
- Dynamic **PCA 2D Cluster Scatter Plot** & **PCA 3D Interactive Space Plot** allowing rotation and interactive exploration.
- Hierarchical linkages visual tree (**Agglomerative Dendrogram**).
- Diagnostic dashboard showcasing the **Elbow Method Curve** (Inertia) and **Silhouette Score Coefficients**.

### 3. Investment Behaviour Intelligence
- Analysis of mortgage loan dependencies across groups.
- Referral channel effectiveness vs. buyer motivations (Sankey and Stacked counts).
- Age group transaction behavior trends.

### 4. Geographic Market Intelligence
- Horizontal country concentrations and regional buyer hotspots.
- Global Sunburst mapping distributions.

### 5. Advanced Visual Analytics Lab
An interactive playground hosting a complete catalog of **all 30 visualizations**! Select any chart from the dropdown to render it dynamically with explanations.

### 6. AI Insight Recommendation Center
- **Key Findings**: Automatic summaries based on active data percentages.
- **Anomalies**: Tracking dissatisfied buyers or abnormal corporate loan structures.
- **Strategic Observations**: Bulleted strategic recommendations for property developers.
- **Persona Mapping**: Clear summaries of the 4 dynamic personas (Corporate Buyers, First-Time Buyers, Luxury Investors, Global Investors).

---

## 🔬 Core Machine Learning Workflow

### 1. Data Prep (`src/preprocessing.py`)
- Removes duplicates, parses dates, and computes ages.
- Performs **One-Hot Encoding** on categorical features (is_corporate, is_investment, loan_applied).
- Normalizes columns using **StandardScaler** to place all coordinates on an equal scale (Mean = 0, Std Dev = 1).

### 2. Unsupervised Grouping (`src/clustering.py`)
- We test multiple candidates for $K$ and plot elbow curves and silhouette ratios.
- **K=4** represents the mathematically optimal cluster count for client segmentation.
- We execute both **K-Means** (centroid-focused) and **Hierarchical Agglomerative Clustering** (Ward linkage tree) using 4 clusters.

---

Enjoy exploring this premium data intelligence console! Open `notebooks/buyer_segmentation.ipynb` to review the math, or run `streamlit run app.py` to start interacting with the charts!
