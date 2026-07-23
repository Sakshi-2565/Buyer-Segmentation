"""
Dashboard Utilities
===================
Provides shared layout helpers, KPI statistic calculators, and CSS injector
utilities to format the Streamlit application as a high-end Business Intelligence platform.
"""

import streamlit as st
import pandas as pd
import numpy as np

def load_css(css_file_path):
    """
    Reads the raw css style sheet and injects it into Streamlit's HTML stream.
    """
    try:
        with open(css_file_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Could not load custom styles from '{css_file_path}'! Falling back to defaults.")

def render_kpi_cards(df):
    """
    Computes key performance indicators from the active DataFrame and displays
    them in sleek, custom modern metric blocks.
    """
    if len(df) == 0:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Buyers Loaded", "0")
        c2.metric("Average Age", "N/A")
        c3.metric("Avg Satisfaction", "N/A")
        c4.metric("Loan Rate", "N/A")
        return
        
    total_buyers = len(df)
    avg_age = df['age'].mean()
    avg_sat = df['satisfaction_score'].mean()
    loan_pct = (df['loan_applied'] == 'Yes').mean() * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Active Cohort", f"{total_buyers:,}", help="Total buyer accounts within active selections.")
        
    with col2:
        st.metric("Mean Buyer Age",f"{avg_age:.1f} Yrs",help="Arithmetic average of buyer demographics.")
    with col3:
        st.metric("Satisfaction Score", f"{avg_sat:.2f} / 5", help="Sentiment survey indices average.")
        
    with col4:
        st.metric("Mortgage Loan Rate", f"{loan_pct:.1f}%", help="Percentage of buyers utilizing bank debt.")

def render_decision_intelligence_kpi(df):
    """
    Computes profitability and financial opportunity KPI metric cards.
    """
    if len(df) == 0:
        return
        
    has_financials = 'total_spend' in df.columns
    total_capital = df['total_spend'].sum() if has_financials else 0
    avg_capital = df['total_spend'].mean() if has_financials else 0
    total_units = df['properties_count'].sum() if has_financials else len(df)
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        if has_financials:
            st.metric("Total Capital Volume", f"${total_capital:,.2f}", help="Sum of financial transactions.")
        else:
            st.metric("Transactions Volume", f"{total_units:,}", help="Total purchases processed.")
        
    with c2:
        if has_financials:
            st.metric("Mean Capital Allocation", f"${avg_capital:,.2f}", help="Average buyer spend portfolio value.")
        else:
            st.metric("Survey Participation", f"{(df['satisfaction_score'].count()):,}", help="Total feedback loops collected.")
        
    with c3:
        if has_financials:
            st.metric("Total Properties Sold", f"{total_units:,} Units", help="Sum of property listings successfully sold.")
        else:
            st.metric("Active Referral Paths", f"{len(df['referral_channel'].unique())}", help="Diversity of acquisition channels.")
