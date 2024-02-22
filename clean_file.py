import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

def clean_df(df):
    mask=(df!='Not Provided')&(df!='Not provided')
    df=df[mask]
    df['EmployeeName']=df['EmployeeName'].apply(lambda x:str(x))
    df['EmployeeName']=df['EmployeeName'].apply(lambda x:x.title())
    df=df.drop(columns=['Notes', 'Id', 'Agency'])
    df['Benefits']=(df['TotalPayBenefits']-df['TotalPay'])
    df=df.dropna(subset=['BasePay', 'EmployeeName'])
    for col in ['BasePay', 'OvertimePay', 'OtherPay']:
        df[col]=df[col].astype(float)
    df=df.reset_index(drop=True)
    return df

def rus_df(df):
    df_clean=df.dropna()
    X=df_clean.drop('Status_Enc', axis=1)
    y=df_clean['Status_Enc']
    rus=RandomUnderSampler()
    X_res, y_res=rus.fit_resample(X,y)
    df_clean=pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name='Status_Enc')], axis=1)
    df_num=df_clean.drop(columns=['Status', 'EmployeeName', 'JobTitle', 'Year'])
    return df_num
    