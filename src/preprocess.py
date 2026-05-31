import pandas as pd
def cleaning(df):
    
    # Refer tge noetbook for cleaning rational

    df_clean = df.copy() # Copy for reproducibility
    
    # Data Hygiene
    df_clean.columns=df_clean.columns.str.replace(" ","")
    
    # Missing Value
    df_clean.dropna(subset = ["CustomerID"],inplace=True)
    df_clean['CustomerID']=df_clean['CustomerID'].astype(int).astype(str)
    df_clean['Description'] = df_clean['Description'].fillna('unknown')

    # Anomalies deletion
    anomalies = (df_clean['Invoice'].astype(str).str.startswith('C',na=False)) | \
                (df_clean['Quantity']<=0) | \
                (df_clean['Price']<=0)
    df_clean = df_clean[~anomalies]

    # De-Duplication
    df_clean = df_clean.sort_values(by=['InvoiceDate']).\
                        drop_duplicates(subset=['Invoice','StockCode',
                                                'Quantity','InvoiceDate',
                                                'Price','CustomerID',
                                                'Country'],keep='last')
    # Feature Engineering
    df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['Price']
    df_clean["is_UK"]=df_clean["Country"].map(lambda x: 1 if x=="United Kingdom" else 0)

    # Validation
    print("\nRows dropped after cleaning:")
    print(f"{(len(df) - len(df_clean)) * 100 / len(df):.2f}%")

    print("\nOverview:")
    df_clean.info()
    
    # Re-index after row deletion
    return df_clean.reset_index(drop=True)
    
