# 🏢 Real Estate Buyer Intelligence & Segmentation Platform
An institutional-grade **Unsupervised Machine Learning & Behavioral Intelligence Dashboard** built with Python, Streamlit, Scikit-Learn, and Plotly. This platform aggregates real estate client transaction records and property portfolios to uncover high-value buyer personas, geographic market concentration, investment behavior, and financial leverage trends.
---
## 🌟 Key Features & Dashboard Consoles
The platform features 6 analytical consoles accessible via an interactive sidebar:
### 1. 📈 Executive Overview
* **Profitability & Decision Intelligence KPIs**: Total Capital Volume, Mean Allocation, and Units Sold.
* **Buyer Cohort Summary**: Dynamic distribution across AI-identified buyer clusters.
* **Market Treemap**: Hierarchical visualization (`Country` → `Region` → `Client Type`).
### 2. 🏢 Buyer Segmentation Analytics Lab
* **PCA Dimensionality Reduction**: Interactive 2D and 3D Principal Component projection spaces.
* **Hierarchical Dendrogram**: Agglomerative clustering visualization using Ward distance metric.
* **Model Diagnostics**: Elbow method (Inertia curve) and Silhouette Coefficient evaluation for optimal $K$ selection.
### 3. 📊 Investment Behaviour Intelligence
* **Financing & Leverage**: Mortgage loan application rates broken down by buyer clusters.
* **Acquisition Motivations**: Ratio of yield-seeking investment property purchases vs. primary residential acquisitions.
* **Referral Channel Conversion**: Multi-channel analysis (Website, Broker Agency, Direct Referrals).
### 4. 🌍 Geographic Market Intelligence
* **Territory Heatmaps**: Country-wise buyer concentration and regional hotspot density maps.
* **Demographic Sunburst**: Concentric hierarchical flow maps (`Country` → `Client Type` → `Acquisition Purpose`).
### 5. 🔬 Advanced Visual Analytics Playground
Explore a catalog of 29+ interactive charts, including:
* **Sankey Flow Diagrams**: Channel $\to$ Client Type $\to$ Cluster mapping.
* **Polar Radar Charts**: Multi-dimensional behavioral profile signatures.
* **Parallel Coordinates Plot**: Scaled feature distribution comparison.
* **Violin & Boxplots**: Age and satisfaction score distribution across cohorts.
* **Scatter Matrices**: Multi-feature correlation grids.
### 6. 💡 AI Insight Recommendation Center
* **Automated Natural Language Findings**: Statistical key findings, market sentiment summaries, and hidden trends.
* **Risk & Anomaly Detection**: Identification of dissatisfied buyer segments and unusual corporate debt leverage.
* **Strategic Growth Recommendations**: Actionable advice for digital channels and institutional portfolio structures.
* **Corporate Target Strategy & Persona Mapping**: Automated centroid persona cards with core buyer characteristics.
---
## 🏗️ Project Architecture & File Structure
```
Segmentation/
│
├── stream.py                  # Main Streamlit web application & UI navigation layout
├── visualization.py           # Plotly, Seaborn, & Matplotlib interactive chart engine
├── insights.py                # Natural Language Automated Business Insights & Anomaly Detector
├── dashboard_utils.py         # Custom CSS injector & sleek KPI card components
├── Feature_Engineering.py     # Encoding, StandardScaler, One-Hot Encoding, & PCA pipeline
├── Models.py                  # K-Means, Agglomerative Clustering, Silhouette & Elbow evaluation
├── preprocessing.py           # Raw dataset loader, cleaning, property aggregation, & merger
│
├── assets/
│   └── css_styles.css         # Dark-mode BI styling and glassmorphism UI theme
│
└── data/
    ├── raw/                   # Raw source CSV files (clients.csv, properties.csv)
    └── buyers_dataset.csv     # Merged and processed dataset output
```
---
## 🛠️ Technology Stack
* **Core & UI Framework**: Python 3.9+, [Streamlit](https://streamlit.io/)
* **Data Processing & Analytics**: Pandas, NumPy
* **Machine Learning**: Scikit-Learn (K-Means, Agglomerative Clustering, PCA, DecisionTrees, StandardScaler)
* **Visualization Engines**: Plotly Express, Plotly Graph Objects, Seaborn, Matplotlib, SciPy (Hierarchical Linkage)
---
## 🚀 Getting Started
### 1. Prerequisites
Ensure you have Python 3.9 or higher installed.
### 2. Installation
Clone the repository and install the required dependencies:
```bash
pip install streamlit pandas numpy scikit-learn plotly matplotlib seaborn scipy
```
### 3. Data Preprocessing Pipeline
Place your raw source files (`clients.csv` and `properties.csv`) under `data/raw/`. Run the preprocessing script to clean and merge the datasets:
```bash
python preprocessing.py
```
### 4. Launching the Platform
Start the Streamlit web dashboard locally:
```bash
streamlit run stream.py
```
The application will open automatically in your browser at `http://localhost:8501`.
---
## 💡 Machine Learning Pipeline Overview
1. **Preprocessing & Feature Engineering**:
   - Cleans currency values, handles missing values, calculates buyer age from date of birth.
   - Aggregates property listing data per buyer (property count, average spend, floor area, property types).
   - Encodes binary features (Label Encoding) and multi-class categories (One-Hot Encoding).
   - Standardizes numerical variables using `StandardScaler`.
   - Compresses high-cardinality features using **Principal Component Analysis (PCA)** preserving 95% variance.
2. **Clustering & Segmentation**:
   - Fits **K-Means Clustering** ($K=4$) based on Elbow and Silhouette diagnostic metrics.
   - Evaluates sub-cohort dendrograms via **Agglomerative Hierarchical Clustering**.
# Buyer-Segmentation
