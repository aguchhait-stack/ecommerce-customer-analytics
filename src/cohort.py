import pandas as pd

def create_cohort(df):

    # Copy to keep to not add intermediate columns
    df_clean = df.copy()

    # CohortMonth
    df_clean['CohortMonth']  =  df_clean.groupby('CustomerID')['InvoiceDate'].transform('min').dt.to_period('M')

    # InvoiceMonth
    df_clean['InvoiceMonth'] =  df_clean['InvoiceDate'].dt.to_period('M')

    # CohortIndex
    df_clean['CohortIndex']  =  (df_clean['InvoiceMonth'].dt.year - df_clean['CohortMonth'].dt.year)*12 +\
                                    (df_clean['InvoiceMonth'].dt.month - df_clean['CohortMonth'].dt.month)
    # Cohort Matrix in wider format
    Cohort_count = df_clean.pivot_table(values = 'CustomerID',index= "CohortMonth",columns= 'CohortIndex',aggfunc='nunique')

    # Cohort sizes
    Cohort_sizes = Cohort_count.iloc[:,0]

    # Retention Percentage
    Retention = Cohort_count.divide(Cohort_sizes,axis=0)
    # Descriptive summary
    print("=="*20)
    print("Cohort Analysis Summary")
    print("=="*20)
    print(f"\n{Retention.iloc[:5,:5].round(2)*100}")
    print(f"\nAverage Month 1 Churn rate: {1-Retention.iloc[:,1].mean():.2%}\n")

    return Retention