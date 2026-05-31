import zipfile
import pandas as pd

# Used context manager to ensure zip and excel file auto-closed

def load_data_from_zip(filepath = 'data/online+retail+ii.zip'):

    with zipfile.ZipFile(filepath,'r') as zf:
        with zf.open('online_retail_II.xlsx') as excel_file:
            df_dict = pd.read_excel(excel_file,sheet_name=None,engine='openpyxl',)
            df = pd.concat(df_dict.values(),axis=0,ignore_index=True)

    print(f"\nSuccessfully loaded {len(df):} transactions")
    return df

