import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

from Feature_Engineering import Encoding
from Models import elbow_silhouette_method, k_means ,agglo_clustering, compute_linkage_matrix
from insights import generate_automated_insights
from dashboard_utils import load_css, render_kpi_cards, render_decision_intelligence_kpi
import visualization as viz
from preprocessing import processing_pipeline

# ------------------------------------------------------------------
# 1. Pipeline Caching & Session Initialization
# ------------------------------------------------------------------


# @st.cache_data
def get_compiled_pipeline_dataset():

    clean_df=processing_pipeline()

    # dataset_path = 'data/buyers_dataset.csv'
    # if not os.path.exists(dataset_path):
    #     st.error(f"Processed dataset buyers_dataset.csv missing under 'data/' directory! Please ensure preprocessing script was run.")
    #     return None, None, None, None    
        
    pca_df, feature_cols = Encoding(clean_df)

    # Train our optimal K-Means model (K = 4 clusters based on diagnostics)
 
    optimal_k = 4
    kmean_model, labels=k_means(pca_df,optimal_k)

    # Insert assignments into base dataframe for reference
    clean_df['Cluster'] = labels
    clean_df['Cluster_Label'] = [f"Cluster {i+1}" for i in labels]

    return clean_df,pca_df,feature_cols,labels

# Load CSS stylesheet and execute data pipeline cache loader
load_css("assets/css_styles.css")
clean_df, scaled_df, feature_cols, cluster_labels = get_compiled_pipeline_dataset()


# ----------------------------------------------------
# 2. Main Page Render and Sidebar Filter Layout
# ----------------------------------------------------
if clean_df is not None:
    
    # Premium Top Dashboard Banner
    st.markdown("""
        <div class="dashboard-header">
            <h1>REAL ESTATE BUYER INTELLIGENCE PLATFORM</h1>
            <p>Institutional Market Segmentation & Investment Behavioral Intelligence Console</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation panel
    st.sidebar.image("https://img.icons8.com/clouds/200/000000/real-estate.png", width=110)
    st.sidebar.markdown("<h3 style='color:#38bdf8;text-align:center;'>PLATFORM NAVIGATION</h3>", unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        "Select Analytics Console",
        [
            "📈 Executive Overview",
            "🏢 Buyer Segmentation Analytics",
            "📊 Investment Behaviour Intelligence",
            "🌍 Geographic Market Intelligence",
            "🔬 Advanced Visual Analytics Lab",
            "💡 AI Insight Recommendation Center"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h4 style='color:#38bdf8;'>ADVANCED BI FILTERS</h4>", unsafe_allow_html=True)
    
    # Dynamic filtration sidebar variables
    # Filter A: Country
    countries = ["All"] + sorted(clean_df['country'].unique().tolist())
    selected_country = st.sidebar.selectbox("Geographic Country", countries)
    
    # Filter B: Region (Dynamic state options based on selected country!)
    if selected_country != "All":
        region_options = clean_df[clean_df['country'] == selected_country]['region'].unique().tolist()
    else:
        region_options = clean_df['region'].unique().tolist()
    regions = ["All"] + sorted(region_options)
    selected_region = st.sidebar.selectbox("Regional Area", regions)
    
    # Filter C: Client Type
    client_types = ["All"] + sorted(clean_df['client_type'].unique().tolist())
    selected_client_type = st.sidebar.selectbox("Client Classification", client_types)
    
    # Filter D: Gender
    genders = ["All"] + sorted(clean_df['gender'].unique().tolist())
    selected_gender = st.sidebar.selectbox("Buyer Gender", genders)
    
    # Filter E: Acquisition Purpose
    purposes = ["All"] + sorted(clean_df['acquisition_purpose'].unique().tolist())
    selected_purpose = st.sidebar.selectbox("Property Purpose", purposes)
    
    # Filter F: Loan Status
    loan_statuses = ["All"] + sorted(clean_df['loan_applied'].unique().tolist())
    selected_loan_status = st.sidebar.selectbox("Financing Applied", loan_statuses)
    
    # Filter G: Satisfaction Score Range
    min_sat, max_sat = st.sidebar.slider("Satisfaction Score Range", 1, 5, (1, 5))
    
    # Filter H: Cluster Selection
    clusters = ["All"] + [f"Cluster {i+1}" for i in sorted(clean_df['Cluster'].unique())]
    selected_cluster = st.sidebar.selectbox("Buyer Cluster Segment", clusters)
    
    # --- Applying Sidebar Filters onto the Dataframes ---
    filtered_df = clean_df.copy()
    if selected_country != "All":
        filtered_df = filtered_df[filtered_df['country'] == selected_country]
    if selected_region != "All":
        filtered_df = filtered_df[filtered_df['region'] == selected_region]
    if selected_client_type != "All":
        filtered_df = filtered_df[filtered_df['client_type'] == selected_client_type]
    if selected_gender != "All":
        filtered_df = filtered_df[filtered_df['gender'] == selected_gender]
    if selected_purpose != "All":
        filtered_df = filtered_df[filtered_df['acquisition_purpose'] == selected_purpose]
    if selected_loan_status != "All":
        filtered_df = filtered_df[filtered_df['loan_applied'] == selected_loan_status]
    filtered_df = filtered_df[(filtered_df['satisfaction_score'] >= min_sat) & (filtered_df['satisfaction_score'] <= max_sat)]
    if selected_cluster != "All":
        filtered_df = filtered_df[filtered_df['Cluster_Label'] == selected_cluster]
        
    # Align scaled vectors with matching filtration indices
    filtered_scaled = scaled_df.loc[filtered_df.index]
    filtered_labels = filtered_df['Cluster'].to_numpy()
    
    # Display the metric cards bar dynamically responding to active filters
    render_kpi_cards(filtered_df)
    
    # ----------------------------------------------------
    # PAGE 1: Executive Overview
    # ----------------------------------------------------
    if page == "📈 Executive Overview":
        st.subheader("📈 Platform Executive Overview")
        
        # Display advanced profitability cards
        st.markdown("### Profitability & Asset Decision Intelligence")
        render_decision_intelligence_kpi(filtered_df)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🏢 Dynamic Real Estate Segment Summary")
            st.write(": Current cohort composition across the 4 core AI-identified buyer cohorts:")
            # Display Cluster distribution bar chart
            fig_dist = viz.plot_cluster_dist(filtered_labels, use_plotly=True)
            st.plotly_chart(fig_dist, use_container_width=True)
            
        with col2:
            st.subheader("🔍 Automated Platform Insights")
            # Fetch dynamic insights list
            findings, trends, anomalies, recs = generate_automated_insights(filtered_df, filtered_labels)
            
            if len(findings) > 0:
                st.markdown("##### 📌 Key Findings & Behavior Summary")
                for f in findings[:2]:
                    st.markdown(f"- {f}")
                st.markdown("##### 🔍 Hidden Demographics & Financial Trends")
                for t in trends[:2]:
                    st.markdown(f"- {t}")
            else:
                st.info("Adjust your filters to discover dynamic cohort patterns.")
            
        # Global Treemap layout
        st.markdown("#### Market Structure Treemap (Country → Region → Client Type)")
        fig_tree = viz.plot_treemap_hierarchy(filtered_df)
        st.plotly_chart(fig_tree, use_container_width=True)
        
    # ----------------------------------------------------
    # PAGE 2: Buyer Segmentation Analytics
    # ----------------------------------------------------
    elif page == "🏢 Buyer Segmentation Analytics":
        st.subheader("🏢 Unsupervised Machine Learning Lab")
        st.write("Explore clustering quality, multi-feature diagnostics, and 2D/3D Principal Component projection spaces.")
        
        tab1, tab2, tab3 = st.tabs(["⚡ PCA 2D/3D Projections", "🌳 Hierarchical Dendrogram", "📊 Performance Diagnostics"])
        
        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### PCA 2D Cluster Space Plot")
                fig_2d = viz.plot_pca_2d(filtered_scaled, filtered_labels, use_plotly=True)
                st.plotly_chart(fig_2d, use_container_width=True)
            with c2:
                st.markdown("#### PCA 3D Interactive Space Plot")
                fig_3d = viz.plot_pca_3d(filtered_scaled, filtered_labels)
                st.plotly_chart(fig_3d, use_container_width=True)
                
        with tab2:
            st.markdown("#### Scipy Hierarchical linkage Tree (Dendrogram)")
            if len(filtered_scaled) < 10:
                st.warning("Insufficient data points loaded to generate hierarchical linkage. Please relax your sidebar filters.")
            else:
                fig_dendro = viz.plot_dendrogram_scipy(filtered_scaled)
                st.pyplot(fig_dendro)
                st.write(
                    "**How to read the Dendrogram**: Individual buyers start at the bottom of the tree as leaves. "
                    "Horizontal brackets represent merges between nodes, and the vertical height shows how dissimilar they were when combined."
                )
                
        with tab3:
            c1, c2 = st.columns(2)
            # Run metrics calculation dynamically for K = 2 to 7
            k_vals, inertias, sil_scores = elbow_silhouette_method(filtered_scaled)
            
            with c1:
                st.markdown("#### The K-Means Elbow Method (Inertia)")
                fig_elbow = viz.plot_elbow_curve(k_vals, inertias)
                st.plotly_chart(fig_elbow, use_container_width=True)
            with c2:
                st.markdown("#### Silhouette Score Coefficient comparison")
                fig_sil = viz.plot_silhouette_comparison(k_vals, sil_scores)
                st.plotly_chart(fig_sil, use_container_width=True)
                
    # ----------------------------------------------------
    # PAGE 3: Investment Behaviour Intelligence
    # ----------------------------------------------------
    elif page == "📊 Investment Behaviour Intelligence":
        st.subheader("📊 Investment & Financial Leverage Intelligence")
        st.write("Inspect how mortgage rates, customer channels, and referrals change based on property purchase motivation.")
        
        c1, c2 = st.columns(2)
        with c1:
            # st.markdown('<div class="bi-card">', unsafe_allow_html=True)
            st.markdown("#### Referral Channel Conversions")
            fig_ref = viz.plot_referral_channel_analysis(filtered_df)
            st.plotly_chart(fig_ref, use_container_width=True)
            # st.markdown('</div>', unsafe_allow_html=True)
            
        with c2:
            # st.markdown('<div class="bi-card">', unsafe_allow_html=True)
            st.markdown("#### Age Group Transaction Behavior Trend")
            fig_age_trend = viz.plot_age_group_trend(filtered_df)
            st.plotly_chart(fig_age_trend, use_container_width=True)
            # st.markdown('</div>', unsafe_allow_html=True)
            
        c3, c4 = st.columns(2)
        with c3:
            # st.markdown('<div class="bi-card">', unsafe_allow_html=True)
            st.markdown("#### Mortgage Loan Application Rates")
            fig_loans = viz.plot_loan_usage_by_cluster(filtered_df, filtered_labels)
            st.plotly_chart(fig_loans, use_container_width=True)
            # st.markdown('</div>', unsafe_allow_html=True)
            
        with c4:
            # st.markdown('<div class="bi-card">', unsafe_allow_html=True)
            st.markdown("#### Global Acquisition Purposes")
            fig_pie = viz.plot_acquisition_purpose_dist(filtered_df)
            st.plotly_chart(fig_pie, use_container_width=True)
            # st.markdown('</div>', unsafe_allow_html=True)
            
    # ----------------------------------------------------
    # PAGE 4: Geographic Market Intelligence
    # ----------------------------------------------------
    elif page == "🌍 Geographic Market Intelligence":
        st.subheader("🌍 Geographic Market Hotspots & Hierarchies")
        st.write("Understand buyer native territories and regional densities using sunburst maps and horizontal matrices.")
        
        c1, c2 = st.columns(2)
        with c1:
            # st.markdown('<div class="bi-card">', unsafe_allow_html=True)
            st.markdown("#### Country Concentration")
            fig_geo = viz.plot_country_heatmap(filtered_df)
            st.plotly_chart(fig_geo, use_container_width=True)
            # st.markdown('</div>', unsafe_allow_html=True)
            
        with c2:
            # st.markdown('<div class="bi-card">', unsafe_allow_html=True)
            st.markdown("#### Regional Concentration (Top 10)")
            fig_reg = viz.plot_geographic_distribution(filtered_df)
            st.plotly_chart(fig_reg, use_container_width=True)
            # st.markdown('</div>', unsafe_allow_html=True)
            
        # st.markdown('<div class="bi-card">', unsafe_allow_html=True)
        st.markdown("#### Sunburst Demographic Map (Country → Client Type → Purpose)")
        fig_sun = viz.plot_sunburst_hierarchy(filtered_df)
        st.plotly_chart(fig_sun, use_container_width=True)
        # st.markdown('</div>', unsafe_allow_html=True)
        
    # ----------------------------------------------------
    # PAGE 5: Advanced Visual Analytics Lab
    # ----------------------------------------------------
    elif page == "🔬 Advanced Visual Analytics Lab":
        st.subheader("🔬 Visual Analytics Playground & Catalog")
        st.write("Select from a premium catalog of advanced data charts representing core customer metrics.")
        
        chart_selection = st.selectbox(
            "Select Interactive Analytical Chart",
            [
                "1. Cluster Distribution Plot",
                "2. PCA 2D Cluster Visualization",
                "3. PCA 3D Interactive Cluster Plot",
                "7. Age vs Satisfaction Scatter by Cluster",
                "8. Loan Usage by Cluster",
                "9. Acquisition Purpose Distribution",
                "10. Country-wise Buyer Concentration Heatmap",
                "11. Region vs Investment Purpose Heatmap",
                "12. Referral Channel Analysis",
                "13. Gender vs Buyer Type Analysis",
                "14. Correlation Heatmap",
                "15. Cluster Feature Importance Plot",
                "16. Client Type Comparison",
                "17. Top 10 Regional Buyer Hotspots",
                "18. Survey Satisfaction Score Distribution",
                "19. Mortgage Loan Dependency Trend by Age",
                "20. Investor Segment Behavior Radar Chart",
                "21. Bubble Chart: Country vs Satisfaction vs Loans",
                "22. Sunburst Chart: Country -> Client Type -> Purpose",
                "23. Treemap Portfolio Density: Country -> Region -> Client Type",
                "24. Sankey Flow: Referral -> Client Type -> Cluster",
                "25. Scaled Feature Parallel Coordinates Plot",
                "26. Age Density Distribution Violin Plot by Cluster",
                "27. Satisfaction Score Boxplot Distribution by Cluster",
                "28. Pairwise Scatter Matrix Grid Relationships",
                "29. Age Group Transaction Behavior Trend"
            ]
        )
        
        # st.markdown('<div class="bi-card">', unsafe_allow_html=True)
        
        # Display the matching chart dynamically
        if chart_selection.startswith("1. "):
            st.plotly_chart(viz.plot_cluster_dist(filtered_labels, use_plotly=True), use_container_width=True)
            st.info("🔍 Counts denote the physical sizes of buyer cohorts grouped dynamically.")
        elif chart_selection.startswith("2. "):
            st.plotly_chart(viz.plot_pca_2d(filtered_scaled, filtered_labels, use_plotly=True), use_container_width=True)
            st.info("🔍 PCA 2D shows structural boundaries. Clear spacing represents clean separations.")
        elif chart_selection.startswith("3. "):
            st.plotly_chart(viz.plot_pca_3d(filtered_scaled, filtered_labels), use_container_width=True)
            st.info("🔍 Multi-dimensional PCA 3D scatter lets you rotate and check data clusters interactively.")
        elif chart_selection.startswith("7. "):
            st.plotly_chart(viz.plot_age_vs_satisfaction(filtered_df, filtered_labels), use_container_width=True)
            st.info("🔍 Age vs. Satisfaction shows sentiment concentrations across demographic cohorts.")
        elif chart_selection.startswith("8. "):
            st.plotly_chart(viz.plot_loan_usage_by_cluster(filtered_df, filtered_labels), use_container_width=True)
            st.info("🔍 Displays standard financing preferences across clusters.")
        elif chart_selection.startswith("9. "):
            st.plotly_chart(viz.plot_acquisition_purpose_dist(filtered_df), use_container_width=True)
            st.info("🔍 Ratio represents investment assets vs. standard residential transactions.")
        elif chart_selection.startswith("10. "):
            st.plotly_chart(viz.plot_country_heatmap(filtered_df), use_container_width=True)
            st.info("🔍 Identifies top buyer locations by size.")
        elif chart_selection.startswith("11. "):
            st.plotly_chart(viz.plot_region_vs_purpose_heatmap(filtered_df), use_container_width=True)
            st.info("🔍 Correlates geographical territory directly with purchase intent density.")
        elif chart_selection.startswith("12. "):
            st.plotly_chart(viz.plot_referral_channel_analysis(filtered_df), use_container_width=True)
            st.info("🔍 Pinpoints marketing and sales channel conversions.")
        elif chart_selection.startswith("13. "):
            st.plotly_chart(viz.plot_gender_vs_buyertype(filtered_df), use_container_width=True)
            st.info("🔍 Cross-tabs buyer entities against client genders.")
        elif chart_selection.startswith("14. "):
            st.plotly_chart(viz.plot_correlation_heatmap(filtered_scaled), use_container_width=True)
            st.info("🔍 Standard Pearson correlation highlights variables that change proportionally.")
        elif chart_selection.startswith("15. "):
            st.plotly_chart(viz.plot_feature_importance(filtered_scaled, filtered_labels), use_container_width=True)
            st.info("🔍 Classification Importance plots exactly which variables drive segment assignments (Age vs. is_investment, etc.).")
        elif chart_selection.startswith("16. "):
            st.plotly_chart(viz.plot_client_type_comparison(filtered_df), use_container_width=True)
            st.info("🔍 Compares individual home buyers vs. corporate clients.")
        elif chart_selection.startswith("17. "):
            st.plotly_chart(viz.plot_geographic_distribution(filtered_df), use_container_width=True)
            st.info("🔍 States/regions showing maximum buy-side activities.")
        elif chart_selection.startswith("18. "):
            st.plotly_chart(viz.plot_satisfaction_distribution(filtered_df), use_container_width=True)
            st.info("🔍 Visual index of satisfaction score distributions.")
        elif chart_selection.startswith("19. "):
            st.plotly_chart(viz.plot_loan_dependency_trends(filtered_df), use_container_width=True)
            st.info("🔍 Loan applied rates generally decline in older segments as capital wealth increases.")
        elif chart_selection.startswith("20. "):
            st.plotly_chart(viz.plot_radar_behavior(filtered_scaled, filtered_labels), use_container_width=True)
            st.info("🔍 Polar coordinate dimensions show signature traits for all clusters at a glance.")
        elif chart_selection.startswith("21. "):
            st.plotly_chart(viz.plot_bubble_country_satisfaction(filtered_df), use_container_width=True)
            st.info("🔍 Visualizes three parameters: Country average satisfaction, loan requirements, and cohort sizes.")
        elif chart_selection.startswith("22. "):
            st.plotly_chart(viz.plot_sunburst_hierarchy(filtered_df), use_container_width=True)
            st.info("🔍 Concentric circles represent proportions of categories grouped hierarchically.")
        elif chart_selection.startswith("23. "):
            st.plotly_chart(viz.plot_treemap_hierarchy(filtered_df), use_container_width=True)
            st.info("🔍 Block area reflects transaction volumes mapped within regional nests.")
        elif chart_selection.startswith("24. "):
            st.plotly_chart(viz.plot_sankey_diagram(filtered_df, filtered_labels), use_container_width=True)
            st.info("🔍 Flows map source categories to target destinations, visualizing paths cleanly.")
        elif chart_selection.startswith("25. "):
            st.plotly_chart(viz.plot_parallel_coordinates(filtered_scaled, filtered_labels), use_container_width=True)
            st.info("🔍 Highlights patterns in scaled values across different clusters.")
        elif chart_selection.startswith("26. "):
            st.plotly_chart(viz.plot_violin_age_by_cluster(filtered_df, filtered_labels), use_container_width=True)
            st.info("🔍 Combines boxplots and probability densities to show age structures.")
        elif chart_selection.startswith("27. "):
            st.plotly_chart(viz.plot_boxplots_numerical(filtered_df, filtered_labels), use_container_width=True)
            st.info("🔍 Displays satisfaction score medians, quartiles, and outliers.")
        elif chart_selection.startswith("28. "):
            st.plotly_chart(viz.plot_pairplot(filtered_scaled, filtered_labels), use_container_width=True)
            st.info("🔍 Scatter matrices represent multi-feature correlations.")
        elif chart_selection.startswith("29. "):
            st.plotly_chart(viz.plot_age_group_trend(filtered_df), use_container_width=True)
            st.info("🔍 Highlights purchase behaviors across specific age cohorts.")
            
        # st.markdown('</div>', unsafe_allow_html=True)
        
    # ----------------------------------------------------
    # PAGE 6: AI Insight Recommendation Center
    # ----------------------------------------------------
    elif page == "💡 AI Insight Recommendation Center":
        st.subheader("💡 AI Automated Insight Recommendation Center")
        st.write(
            "Our integrated Natural-Language Insights Engine dynamically audits active profiles to construct "
            "actionable business observations, identify anomalies, and recommend developer growth programs."
        )
        
        findings, trends, anomalies, recs = generate_automated_insights(filtered_df, filtered_labels)
        
        # Display insights cleanly inside card blocks
        col1, col2 = st.columns(2)
        
        with col1:
            # st.markdown('<div class="bi-card">', unsafe_allow_html=True)
            st.markdown("#### 🔍 Active Core Key Findings")
            if len(findings) > 0:
                for f in findings:
                    st.markdown(f"<div class='alert-box alert-info'>{f}</div>", unsafe_allow_html=True)
            else:
                st.write("No findings generated.")
            # st.markdown('</div>', unsafe_allow_html=True)
            
            # st.markdown('<div class="bi-card">', unsafe_allow_html=True)
            st.markdown("#### ⚠️ Transaction Risk & Anomalies")
            if len(anomalies) > 0:
                for a in anomalies:
                    st.markdown(f"<div class='alert-box alert-warning'>{a}</div>", unsafe_allow_html=True)
            else:
                st.success("No anomalies or critical transaction risks detected.")
            # st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            # st.markdown('<div class="bi-card">', unsafe_allow_html=True)
            st.markdown("#### 📈 Hidden Consumer Trends")
            if len(trends) > 0:
                for t in trends:
                    st.markdown(f"<div class='alert-box alert-info'>{t}</div>", unsafe_allow_html=True)
            else:
                st.write("No trends detected.")
            # st.markdown('</div>', unsafe_allow_html=True)
            
            # st.markdown('<div class="bi-card">', unsafe_allow_html=True)
            st.markdown("#### 🚀 Strategic Growth Recommendations")
            if len(recs) > 0:
                for r in recs:
                    st.markdown(f"<div class='alert-box alert-success'>{r}</div>", unsafe_allow_html=True)
            else:
                st.write("No recommendations generated.")
            # st.markdown('</div>', unsafe_allow_html=True)
            
        # Target Corporate Campaigns section
        # st.markdown('<div class="bi-card">', unsafe_allow_html=True)
        st.markdown("#### Corporate Target Strategy & Persona Mapping")
        st.write("Below are the strategic client personas automatically generated by analyzing the baseline data centroids:")
        
        cluster_ids = sorted(clean_df['Cluster'].unique())
        for cid in cluster_ids:
            cdf = clean_df[clean_df['Cluster'] == cid]
            top_purpose = cdf['acquisition_purpose'].mode()[0] if not cdf['acquisition_purpose'].empty else "Investment"
            top_client = cdf['client_type'].mode()[0] if not cdf['client_type'].empty else "Individual"
            avg_sat = cdf['satisfaction_score'].mean()
            avg_age = cdf['age'].mean()
            loan_rate = (cdf['loan_applied'] == 'Yes').mean() * 100
            
            persona_title = f"{top_client} {top_purpose} Cohort"
            persona_desc = f"Cluster {cid+1} consists of {len(cdf):,} buyers with an average age of {avg_age:.1f} years and satisfaction rating of {avg_sat:.2f}/5."
            chars = [
                f"Primary Acquisition Purpose: {top_purpose}",
                f"Client Classification: {top_client}",
                f"Mortgage Loan Utilization: {loan_rate:.1f}%",
                f"Total Cohort Size: {len(cdf):,} accounts"
            ]
            
            st.markdown(f"""
                <div class="persona-card">
                    <div class="persona-header">Cluster {cid+1}: {persona_title}</div>
                    <div class="persona-desc">{persona_desc}</div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("**Core Characteristics:**")
            for char in chars:
                st.markdown(f"- {char}")
            st.markdown("---")
        # st.markdown('</div>', unsafe_allow_html=True)
else:
    st.warning("Please confirm pipeline compilation and buyers_dataset.csv availability.")


