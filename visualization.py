import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from scipy.cluster.hierarchy import dendrogram, linkage

# Global color theme overrides for dark mode BI integration
DARK_TEMPLATE = "plotly_dark"
COLOR_SEQUENCE = px.colors.qualitative.Dark24
MATRIX_COLORSCALE = "Bluyl"

# ----------------------------------------------------
# SECTION 1: Clustering Evaluation & Diagnostics (1-6)
# ----------------------------------------------------

def plot_cluster_dist(labels, use_plotly=True):
    counts = pd.Series(labels).value_counts().sort_index()
    dist_df = pd.DataFrame({'Cluster': [f"Cluster {i+1}" for i in counts.index], 'Count': counts.values})
    
    if use_plotly:
        fig = px.bar(
            dist_df, x='Cluster', y='Count', color='Cluster', text_auto=True,
            title='<b>1. Cluster Buyer Account Distribution</b>',
            color_discrete_sequence=COLOR_SEQUENCE, template=DARK_TEMPLATE
        )
        fig.update_layout(showlegend=False, margin=dict(t=50, b=50))
        return fig
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=dist_df, x='Cluster', y='Count', palette='viridis', ax=ax)
        ax.set_title("1. Cluster Buyer Account Distribution")
        return fig
    

    

def plot_pca_2d(scaled_df, labels, use_plotly=True):
    pca = PCA(n_components=2, random_state=42)
    pca_data = pca.fit_transform(scaled_df)
    pca_df = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = [f"Cluster {i+1}" for i in labels]
    
    if use_plotly:
        fig = px.scatter(
            pca_df, x='PC1', y='PC2', color='Cluster',
            title='<b>2. PCA 2D Cluster Visualization</b>',
            color_discrete_sequence=COLOR_SEQUENCE, template=DARK_TEMPLATE
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='Set1', ax=ax)
        return fig
    



def plot_pca_3d(df, labels):
    pca = PCA(n_components=3, random_state=42)
    pca_data = pca.fit_transform(df)
    pca_df = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2', 'PC3'])
    pca_df['Cluster'] = [f"Cluster {i+1}" for i in labels]
    
    fig = px.scatter_3d(
        pca_df, x='PC1', y='PC2', z='PC3', color='Cluster',
        title='<b>3. PCA 3D Interactive Cluster Plot</b>',
        color_discrete_sequence=COLOR_SEQUENCE, template=DARK_TEMPLATE
    )
    fig.update_layout(margin=dict(t=50, b=50))
    return fig



def plot_dendrogram_scipy(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0b0f19')
    
    linkage_matrix = linkage(df, method='ward')
    dendrogram(
        linkage_matrix, truncate_mode='lastp', p=20,
        show_leaf_counts=True, ax=ax,
        above_threshold_color='#64748b'
    )
    ax.set_title("4. Hierarchical Agglomerative Dendrogram (Top 20 Merges)", color='#f8fafc', fontsize=14, fontweight='bold')
    ax.set_xlabel("Leaf Sub-cohorts", color='#94a3b8')
    ax.set_ylabel("Ward Distance Metric", color='#94a3b8')
    ax.tick_params(colors='#64748b')
    return fig



def plot_elbow_curve(k_values, inertias):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_values, y=inertias, mode='lines+markers',
        line=dict(color='#00b4d8', width=3),
        marker=dict(size=8, color='#f77f00')
    ))
    fig.update_layout(
        title='<b>5. The Elbow Method (Inertia Curve)</b>',
        xaxis_title='Cluster Count (K)', yaxis_title='Inertia / WSS',
        template=DARK_TEMPLATE, margin=dict(t=50, b=50)
    )
    return fig



def plot_silhouette_comparison(k_values, scores):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=k_values, y=scores,
        marker_color='#9d4edd'
    ))
    fig.update_layout(
        title='<b>6. Silhouette Score Comparison Index</b>',
        xaxis_title='Cluster Count (K)', yaxis_title='Average Silhouette Coefficient',
        template=DARK_TEMPLATE, margin=dict(t=50, b=50)
    )
    return fig




def plot_age_vs_satisfaction(df_clients, labels):
    plot_df = df_clients.copy()
    plot_df['Cluster'] = [f"Cluster {i+1}" for i in labels]
    # Add minor jitter to discrete rating values to improve visualization clarity
    plot_df['Satisfaction (Jittered)'] = plot_df['satisfaction_score'] + np.random.uniform(-0.15, 0.15, len(plot_df))
    
    fig = px.scatter(
        plot_df, x='age', y='Satisfaction (Jittered)', color='Cluster',
        title='<b>7. Age vs. Satisfaction Scatter by Cluster Persona</b>',
        color_discrete_sequence=COLOR_SEQUENCE, opacity=0.7, template=DARK_TEMPLATE
    )
    return fig



def plot_loan_usage_by_cluster(df, labels):
    plot_df = df.copy()
    plot_df['Cluster'] = [f"Cluster {i+1}" for i in labels]
    
    fig = px.histogram(
        plot_df, x='Cluster', color='loan_applied',
        title='<b>8. Loan Usage Rate by Cluster Persona</b>',
        barmode='group', color_discrete_sequence=['#ff4d6d', '#3a86c8'],
        template=DARK_TEMPLATE
    )
    return fig



def plot_acquisition_purpose_dist(df):
    fig = px.pie(
        df, names='acquisition_purpose', hole=0.4,
        title='<b>9. Acquisition Purpose Global Distribution</b>',
        color_discrete_sequence=['#00b4d8', '#f77f00'], template=DARK_TEMPLATE
    )
    return fig


def plot_country_heatmap(df):
    counts = df['country'].value_counts().reset_index()
    counts.columns = ['Country', 'Buyers Count']
    
    fig = px.bar(
        counts, x='Buyers Count', y='Country', orientation='h',
        title='<b>10. Country-wise Buyer Concentration Heatmap</b>',
        color='Buyers Count', color_continuous_scale=MATRIX_COLORSCALE,
        template=DARK_TEMPLATE
    )
    return fig

def plot_region_vs_purpose_heatmap(df):
    pivot = df.pivot_table(index='region', columns='acquisition_purpose', values='client_id', aggfunc='count', fill_value=0)
    
    fig = px.imshow(
        pivot, text_auto=True, aspect="auto",
        title='<b>11. Region vs. Investment Purpose Density</b>',
        color_continuous_scale=MATRIX_COLORSCALE, template=DARK_TEMPLATE
    )
    return fig

def plot_referral_channel_analysis(df):
    fig = px.histogram(
        df, x='referral_channel', color='acquisition_purpose',
        title='<b>12. Referral Channel Efficiency & Motivation</b>',
        barmode='stack', color_discrete_sequence=COLOR_SEQUENCE, template=DARK_TEMPLATE
    )
    return fig

# ----------------------------------------------------
# SECTION 3: Demographic & Structural Analytics (13-18)
# ----------------------------------------------------

def plot_gender_vs_buyertype(df):
    fig = px.histogram(
        df, x='gender', color='client_type',
        title='<b>13. Gender vs. Buyer Corporate/Individual Type</b>',
        barmode='group', color_discrete_sequence=['#38bdf8', '#fbbf24'], template=DARK_TEMPLATE
    )
    return fig

def plot_correlation_heatmap(scaled_df):
    corr = scaled_df.corr()
    
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        title='<b>14. Correlation Matrix of Scaled Variables</b>',
        color_continuous_scale="RdBu", template=DARK_TEMPLATE
    )
    return fig

def plot_feature_importance(scaled_df, labels):
    # Fit a simple decision tree classifier to extract feature importances
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(scaled_df, labels)
    
    importance_df = pd.DataFrame({
        'Feature': scaled_df.columns,
        'Importance': clf.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    
    fig = px.bar(
        importance_df, x='Importance', y='Feature', orientation='h',
        title='<b>15. Cluster Classification Feature Importance Plot</b>',
        color='Importance', color_continuous_scale="Viridis", template=DARK_TEMPLATE
    )
    return fig

def plot_client_type_comparison(df):
    fig = px.pie(
        df, names='client_type',
        title='<b>16. Client Type Volume Comparison</b>',
        color_discrete_sequence=['#10b981', '#3b82f6'], template=DARK_TEMPLATE
    )
    return fig

def plot_geographic_distribution(df):
    reg_counts = df['region'].value_counts().head(10).reset_index()
    reg_counts.columns = ['Region', 'Count']
    
    fig = px.bar(
        reg_counts, x='Region', y='Count',
        title='<b>17. Top 10 Regional Buyer Hotspots</b>',
        color='Count', color_continuous_scale="Turbo", template=DARK_TEMPLATE
    )
    return fig

def plot_satisfaction_distribution(df):
    fig = px.histogram(
        df, x='satisfaction_score', nbins=5,
        title='<b>18. Survey Satisfaction Score Distribution</b>',
        color_discrete_sequence=['#14b8a6'], template=DARK_TEMPLATE
    )
    return fig

# ----------------------------------------------------
# SECTION 4: Advanced Analytical BI Charts (19-24)
# ----------------------------------------------------

def plot_loan_dependency_trends(df):
    # Analyze loan applied rate by age groups
    df_copy = df.copy()
    df_copy['AgeGroup'] = pd.cut(df_copy['age'], bins=[18, 30, 45, 60, 75, 100], labels=['18-29', '30-44', '45-59', '60-74', '75+'])
    
    grouped = df_copy.groupby('AgeGroup', observed=False)['loan_applied'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
    grouped.columns = ['Age Group', 'Loan Rate (%)']
    
    fig = px.line(
        grouped, x='Age Group', y='Loan Rate (%)', markers=True,
        title='<b>19. Mortgage Loan Dependency Trend by Age Cohorts</b>',
        template=DARK_TEMPLATE
    )
    fig.update_traces(line=dict(color='#f43f5e', width=3))
    return fig

def plot_radar_behavior(scaled_df, labels):
    df_temp = pd.DataFrame(scaled_df).copy()
    df_temp['Cluster'] = [f"Cluster {i+1}" for i in labels]
    means = df_temp.groupby('Cluster').mean()
    
    fig = go.Figure()
    
    # We display up to 5 variables to keep radar readable
    features = list(means.columns)[:5]
    
    for cluster_name in means.index:
        values = means.loc[cluster_name, features].tolist()
        values.append(values[0])  # Close the radar loop
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=features + [features[0]],
            fill='toself',
            name=cluster_name
        ))
        
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[-1.5, 1.5])
        ),
        title='<b>20. Investor Segment Behavior Radar Chart</b>',
        template=DARK_TEMPLATE
    )
    return fig

def plot_bubble_country_satisfaction(df):
    grouped = df.groupby('country').agg(
        avg_sat=('satisfaction_score', 'mean'),
        loan_rate=('loan_applied', lambda x: (x == 'Yes').mean() * 100),
        size=('client_id', 'count')
    ).reset_index()
    
    fig = px.scatter(
        grouped, x='avg_sat', y='loan_rate', size='size', color='country',
        title='<b>21. Country Bubble: Satisfaction vs. Loan Usage vs. Cohort Size</b>',
        labels={'avg_sat': 'Avg Satisfaction Score', 'loan_rate': 'Loan Applied Rate (%)', 'size': 'Buyer Cohort Size'},
        template=DARK_TEMPLATE
    )
    return fig

def plot_sunburst_hierarchy(df):
    fig = px.sunburst(
        df, path=['country', 'client_type', 'acquisition_purpose'],
        title='<b>22. Demographic Sunburst: Country → Client Type → Purpose</b>',
        color_discrete_sequence=COLOR_SEQUENCE, template=DARK_TEMPLATE
    )
    return fig

def plot_treemap_hierarchy(df):
    fig = px.treemap(
        df, path=['country', 'region', 'client_type'],
        title='<b>23. Treemap Portfolio Density: Country → Region → Client Type</b>',
        color_discrete_sequence=COLOR_SEQUENCE, template=DARK_TEMPLATE
    )
    return fig

def plot_sankey_diagram(df, labels):
    plot_df = df.copy()
    plot_df['Cluster'] = [f"Cluster {i+1}" for i in labels]
    
    # Sankey nodes definition
    # Source: Referral Channel
    # Target 1: Client Type
    # Target 2: Cluster
    
    referrals = sorted(plot_df['referral_channel'].unique().tolist())
    client_types = sorted(plot_df['client_type'].unique().tolist())
    clusters = sorted(plot_df['Cluster'].unique().tolist())
    
    all_nodes = referrals + client_types + clusters
    node_indices = {node: i for i, node in enumerate(all_nodes)}
    
    sources = []
    targets = []
    values = []
    
    # 1. Referral to Client Type link
    g1 = plot_df.groupby(['referral_channel', 'client_type']).size().reset_index(name='count')
    for _, row in g1.iterrows():
        sources.append(node_indices[row['referral_channel']])
        targets.append(node_indices[row['client_type']])
        values.append(row['count'])
        
    # 2. Client Type to Cluster link
    g2 = plot_df.groupby(['client_type', 'Cluster']).size().reset_index(name='count')
    for _, row in g2.iterrows():
        sources.append(node_indices[row['client_type']])
        targets.append(node_indices[row['Cluster']])
        values.append(row['count'])
        
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20, line=dict(color="black", width=0.5),
            label=all_nodes, color="skyblue"
        ),
        link=dict(
            source=sources, target=targets, value=values,
            color="rgba(56, 189, 248, 0.2)"
        )
    )])
    fig.update_layout(title_text="<b>24. Sankey Flow: Referral Channel → Client Type → Cluster Group</b>", template=DARK_TEMPLATE)
    return fig

# ----------------------------------------------------
# SECTION 5: High-Value Analytics Playground (25-30)
# ----------------------------------------------------

def plot_parallel_coordinates(scaled_df, labels):
    # We display a selection of 5 features for visual clarity
    cols = list(scaled_df.columns)[:5]
    df_temp = pd.DataFrame(scaled_df[cols]).copy()
    df_temp['Cluster_ID'] = labels
    
    fig = px.parallel_coordinates(
        df_temp, color="Cluster_ID",
        title="<b>25. Scaled Feature Parallel Coordinates Plot</b>",
        color_continuous_scale=px.colors.diverging.Tealrose,
        template=DARK_TEMPLATE
    )
    return fig

def plot_violin_age_by_cluster(df, labels):
    plot_df = df.copy()
    plot_df['Cluster'] = [f"Cluster {i+1}" for i in labels]
    
    fig = px.violin(
        plot_df, x='Cluster', y='age', color='Cluster', box=True, points="all",
        title="<b>26. Age Density Distribution Violin Plot by Cluster</b>",
        color_discrete_sequence=COLOR_SEQUENCE, template=DARK_TEMPLATE
    )
    return fig

def plot_boxplots_numerical(df, labels):
    plot_df = df.copy()
    plot_df['Cluster'] = [f"Cluster {i+1}" for i in labels]
    
    fig = px.box(
        plot_df, x='Cluster', y='satisfaction_score', color='Cluster',
        title="<b>27. Satisfaction Score Boxplot Distribution by Cluster</b>",
        color_discrete_sequence=COLOR_SEQUENCE, template=DARK_TEMPLATE
    )
    return fig

def plot_pairplot(scaled_df, labels):
    # Generate pairwise relationship scatter matrix using plotly
    cols = list(scaled_df.columns)[:3]  # Choose 3 features to fit grid nicely
    df_temp = pd.DataFrame(scaled_df[cols]).copy()
    df_temp['Cluster'] = [f"Cluster {i+1}" for i in labels]
    
    fig = px.scatter_matrix(
        df_temp, dimensions=cols, color='Cluster',
        title="<b>28. Pairwise Scatter Matrix Grid Relationships</b>",
        color_discrete_sequence=COLOR_SEQUENCE, template=DARK_TEMPLATE
    )
    return fig

def plot_age_group_trend(df):
    df_copy = df.copy()
    df_copy['AgeGroup'] = pd.cut(df_copy['age'], bins=[18, 30, 45, 60, 75, 100], labels=['18-29', '30-44', '45-59', '60-74', '75+'])
    
    counts = df_copy.groupby(['AgeGroup', 'acquisition_purpose'], observed=False).size().reset_index(name='Volume')
    
    fig = px.bar(
        counts, x='AgeGroup', y='Volume', color='acquisition_purpose',
        title="<b>29. Age Group Transaction Behavior Trend</b>",
        barmode='group', color_discrete_sequence=['#06b6d4', '#e11d48'], template=DARK_TEMPLATE
    )
    return fig

def plot_cluster_profile_radar(scaled_df, labels):
    return plot_radar_behavior(scaled_df, labels)  # Radar acts as advanced profile layout
