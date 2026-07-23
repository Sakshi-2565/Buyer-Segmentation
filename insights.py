import pandas as pd
import numpy as np

def generate_automated_insights(df, labels):
    """
    Dynamically scans the filtered/active buyer dataset and computes high-level BI observations.
    
    Returns:
    - findings: List of key statistical findings.
    - trends: List of identified consumer trends.
    - anomalies: List of suspicious outliers or anomalies.
    - recommendations: Strategic recommendation lists.
    """
    findings=[]
    trends =[]
    anomalies=[]
    recommendations =[]
    
    if len(df) == 0:
        return findings, trends, anomalies, recommendations

    # Copy the df to compute stats safely
    insights_df = df.copy()
    insights_df['Cluster'] = labels
    
    # ----------------------------------------------------
    # 1. Key Findings Computations
    # ----------------------------------------------------
    total_buyers = len(insights_df)
    
    # Satisfaction comparison
    avg_sat = insights_df['satisfaction_score'].mean()
    high_sat_rate = (insights_df['satisfaction_score'] >= 4).mean() * 100
    findings.append(f"📊 Market Sentiment: {high_sat_rate:.1f}% of active buyers reported high satisfaction (score of 4 or 5), averaging {avg_sat:.2f}/5.")

    # Investment vs Personal Use
    invest_pct = (insights_df['acquisition_purpose'] == 'Investment').mean() * 100
    findings.append(f"🏢 Acquisition Motivation: {invest_pct:.1f}% of acquisitions represent yield-seeking property investments, while the rest represent personal home purchases.")
    
    # Top countries
    top_country = insights_df['country'].mode()[0] if not insights_df['country'].empty else "Unknown"
    top_country_pct = (insights_df['country'] == top_country).mean() * 100
    findings.append(f"🌍 Geographic Concentration: The primary buyer source is {top_country}, accounting for {top_country_pct:.1f}% of the total transaction ledger.")

    # ----------------------------------------------------
    # 2. Hidden Trends Detection
    # ----------------------------------------------------
    # Younger buyers and loans correlation
    young_mask = insights_df['age'] < 40
    if young_mask.any():
        young_loan_rate = (insights_df[young_mask]['loan_applied'] == 'Yes').mean() * 100
        older_loan_rate = (insights_df[~young_mask]['loan_applied'] == 'Yes').mean() * 100 if (~young_mask).any() else 0
        trends.append(f"📈 Age-Financing Leverage Trend: Younger buyers under 40 display a massive {young_loan_rate:.1f}% dependency on mortgage financing, compared to only {older_loan_rate:.1f}% in older age brackets.")
    
    # High-value spending patterns (if spend columns exist)
    if 'total_spend' in insights_df.columns:
        corp_spend = insights_df[insights_df['client_type'].str.lower().isin(['corporate', 'company'])]['total_spend'].mean()
        ind_spend = insights_df[~insights_df['client_type'].str.lower().isin(['corporate', 'company'])]['total_spend'].mean()
        if not np.isnan(corp_spend) and not np.isnan(ind_spend):
            trends.append(f"💵 Portfolio Capital Trend: Corporate buyers command {(corp_spend/ind_spend if ind_spend > 0 else 1.0):.1f}x higher total capital allocation than individual residential buyers.")
            
    # Referral conversions
    website_pct = (insights_df['referral_channel'] == 'Website').mean() * 100
    agency_pct = (insights_df['referral_channel'] == 'Agency').mean() * 100
    trends.append(f"📣 Channel Trend: Digital acquisition (Website) drives {website_pct:.1f}% of transactions, while classic broker paths (Agency) represent {agency_pct:.1f}%.")

    # ----------------------------------------------------
    # 3. Anomalies Tracking
    # ----------------------------------------------------
    # Dissonant groups: High spend + Low satisfaction
    dissatisfied_buyers = insights_df[insights_df['satisfaction_score'] <= 2]
    if len(dissatisfied_buyers) > 0:
        pct_dissatisfied = (len(dissatisfied_buyers) / total_buyers) * 100
        anomalies.append(f"⚠️ Risk Anomaly: {pct_dissatisfied:.1f}% of property buyers are highly dissatisfied (ratings 1-2). Immediate post-sales follow-ups are recommended.")
        
    # High loan corporate transactions (corporate entities rarely apply for basic residential mortgage loans)
    corp_loan_buyers = insights_df[insights_df['client_type'].str.lower().isin(['corporate', 'company']) & (insights_df['loan_applied'] == 'Yes')]
    if len(corp_loan_buyers) > 0:
        anomalies.append(f"🔍 Debt-Leveraged Entities: Found {len(corp_loan_buyers)} corporate accounts utilizing mortgage loans. Represents an anomaly as companies usually fund acquisitions via commercial credit lines.")

    
    # ----------------------------------------------------
    # 4. Actionable Business Recommendations
    # ----------------------------------------------------
    recommendations.append("🚀 Optimize Digital Channels: Since Web referrals represent a dominant share, deploy user-friendly virtual property tours and automated calculators to further lower acquisition costs.")
    
    if invest_pct > 40:
        recommendations.append("💼 Launch Institutional Portfolios: The strong investment appetite indicates high demand for bulk leasing structures. Offer rental guarantee options to secure high-value luxury investors.")
    else:
        recommendations.append("🏡 Foster Residential Subsidies: High personal use motivation requires smooth mortgage approvals. Build direct partnerships with major banks to streamline loan underwriting pipelines.")
        
    if len(dissatisfied_buyers) > 0:
        recommendations.append("📋 Implement Quality Assurance Protocols: Establish immediate feedback surveys for any score below 3, routing them to high-level client relations personnel to safeguard retention.")
        
    return findings, trends, anomalies, recommendations
