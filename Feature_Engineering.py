import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder ,OneHotEncoder
from sklearn.decomposition import PCA

def Encoding(df):
    # Drop columns not needed for clustering
    df = df.drop(columns=['client_id', 'first_name', 'last_name', 'date_of_birth'])

    #-----------------------------------Encoding---------------------------------------------

    # (A): Label Encode binary columns (2 unique values)

    le = LabelEncoder()
    for col in ['gender', 
                'acquisition_purpose', 
                'loan_applied', 
                'client_type']:
        df[col] = le.fit_transform(df[col].astype(str))


    # 6(B): StandardScaler for all numerical columns ---

    num_cols = [
        'satisfaction_score', 'age', 'num_properties',
        'avg_sale_price', 'total_sale_price', 'avg_floor_area',
        'num_sold', 'num_available', 'num_apartment', 'num_office'
    ]
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(df[num_cols])
    num_df = pd.DataFrame(num_scaled, columns=num_cols, index=df.index)

    # return num_df

    # (B): One-Hot Encode referral_channel (3 unique values: Website, Agency, Client)

    ohe_small = OneHotEncoder(sparse_output=False,drop='first')
    
    rc_encoded = ohe_small.fit_transform(df[['referral_channel','country', 'region']])

    rc_df = pd.DataFrame(rc_encoded, 
                         columns=ohe_small.get_feature_names_out(['referral_channel','country', 'region']),
                         index=df.index)
   

    # 7. Combine Final Feature Matrix

    binary_df = df[['gender', 'acquisition_purpose', 'loan_applied', 'client_type']].reset_index(drop=True)
    
    final_df = pd.concat([
        binary_df,                          # label encoded binary columns
        rc_df.reset_index(drop=True),      # PCA components for country + region
        num_df.reset_index(drop=True)       # scaled numerical columns
    ], axis=1)
    
    print(f"Final feature matrix shape: {final_df.shape}\n")
    #return final_df
    actual_df=final_df.copy()
        
        
 # (C): PCA for high-cardinality columns ---

    # 81 OHE columns → compressed via PCA    
    # Retain 95% of variance — PCA automatically picks the number of components
    pca = PCA(n_components=0.95, random_state=42)
    pca_result = pca.fit_transform(final_df)
    pca_df = pd.DataFrame(
        pca_result,
        columns=[f'pca_{i+1}' for i in range(pca_result.shape[1])],
        index=df.index
        )
    print(f"\nPCA: {final_df.shape[1]} OHE columns → {pca_result.shape[1]} components "
        f"({pca.explained_variance_ratio_.sum():.1%} variance retained)\n")
 
    pca_columns=pca_df.columns
    return pca_df,pca_columns