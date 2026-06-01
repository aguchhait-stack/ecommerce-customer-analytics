import pandas as pd

def create_rfm(df_clean):

        # Snap date
        snap_date = df_clean['InvoiceDate'].max()+pd.Timedelta(days=1)

        # RFM table
        rfm =   df_clean.groupby('CustomerID',as_index=False).agg(
                Recency=("InvoiceDate",lambda x: (snap_date-x.max()).days),
                Frequency = ("Invoice","nunique"),
                Monetary = ("TotalPrice","sum"))
        # Descriptive summary
        print("\n"+"=="*20)
        print("Base RFM Metrics Summary")
        print("=="*20)
        print(f"\n{rfm.describe().round(2)}\n")
        return rfm

