import pandas as pd


def load_data(clients,properties):
    client=pd.read_csv(clients)
    properties=pd.read_csv(properties)
    
    print("Loading data...\n")
    return client,properties

def checking_data(clients,properties):

    # ## INFO
    # info_c=client.info()
    # info_p=properties.info()
    # #return info_c,info_p

    ## ISNULL
    null_c =clients.isnull().sum()
    null_p =properties.isnull().sum()
    null="Checking Null Values of Clients.csv:\n",null_c,"\n Checking Null Values of Properties.csv:\n",null_p

    ## ISDUPLICATE
    dup_c = clients.duplicated().sum()
    dup_p = properties.duplicated().sum()
    return null," Checking Duplicate Values of Clients.csv:",dup_c,"Checking Duplicate Values of Properties.csv:\n",dup_p
    

def processing_properties(properties):

    # Creating Copy

    properties =properties.copy()

    # Clean property sale_price
    properties['sale_price'] = properties['sale_price'].str.replace('$', '').str.replace(',', '').astype(float)

    
    int_cols=['unit_number','floor_area_sqft','sale_price']

    for col in int_cols:
        if col in properties.columns:
            mean=properties[col].mean()
            properties[col]=properties[col].fillna(mean)

    print("Extracting $ from sales_price and Filling NULL values....\n")

    return properties


def aggregating_properties(properties):

    props_agg = properties.groupby('client_ref').agg(
    num_properties   = ('listing_id',      'count'),
    avg_sale_price   = ('sale_price',      'mean'),
    total_sale_price = ('sale_price',      'sum'),
    avg_floor_area   = ('floor_area_sqft', 'mean'),
    num_sold         = ('listing_status',  lambda x: (x == 'Sold').sum()),
    num_available    = ('listing_status',  lambda x: (x == 'Available').sum()),
    num_apartment    = ('unit_category',   lambda x: (x == 'Apartment').sum()),
    num_office       = ('unit_category',   lambda x: (x == 'Office').sum()),).reset_index().rename(columns={'client_ref': 'client_id'})

    print("Aggregating Properties Data....\n")

    return props_agg
 


def processing_clients(clients):

    # Creating Copy of dataset

    clients = clients.copy()



    # Age from Date of Birth

    clients['date_of_birth'] = pd.to_datetime(clients['date_of_birth'], dayfirst=True, errors='coerce')
    clients['age'] = ((pd.Timestamp('today') - clients['date_of_birth']).dt.days / 365)
    clients['age'] = clients['age'].fillna(clients['age'].median())  # fill missing with median

    print("DOB -> Age...\n")
 
    # For categorical columns, fill with the mode (most common value) or 'Unknown'

    cat_cols = ['client_type', 'gender', 'country', 'region', 'acquisition_purpose', 'loan_applied', 'referral_channel']
    int_cols = ['age','satisfaction_score']

    for col in cat_cols:
        if col in clients.columns:
            mode_val = clients[col].mode()[0]
            clients[col] = clients[col].fillna(mode_val)
            # Ensure text is clean and uniform (e.g. stripped of extra spaces, title cased)
            clients[col] = clients[col].astype(str).str.strip()

    for col in int_cols:
        if col in clients.columns:
            mean=clients[col].mean()
            clients[col]=clients[col].fillna(mean)
    print("Filling NULL values...\n")

    return clients
    

def merging_client_prop(clients,properties):
    
    df = clients.merge(properties, on='client_id', how='inner')
    print("Merging client.csv and properties.csv ....\n")
    print(f"Merged shape: {df.shape}")
    clean_df=pd.DataFrame(df)
    clean_df.to_csv('data/buyers_dataset.csv',index=False)
    
    return clean_df


client_path='data/raw/clients.csv'
properties_path='data/raw/properties.csv'

def processing_pipeline(client_path=client_path,properties_path=properties_path):

    df_client,df_properties=load_data(client_path,properties_path)
    #print(load_data)

    cd=checking_data(df_client,df_properties)
    #print(cd)

    df_props=processing_properties(df_properties)
    #print(df_props)

    df_props=aggregating_properties(df_props)
    #print(df_props)

    df_clients=processing_clients(df_client)
    #print(df_client)

    clean_df=merging_client_prop(df_clients,df_props)
    #print(df)

    return clean_df
    
