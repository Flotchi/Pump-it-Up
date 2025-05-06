import pandas as pd
import numpy as np

def clean_data(df):

    """Cleans water pump data by handling missing values, removing irrelevant columns,
    and creating new features."""

    #remove duplicates and NA
    #df = df.drop_duplicates()
    #df.dropna(thresh=35, inplace=True)

    #drop irrelevant columns
    columns_to_drop = [
        'wpt_name', 'num_private', 'recorded_by', 'subvillage', 'lga',
        'region_code', 'management_group', 'extraction_type_group',
        'extraction_type_class', 'payment', 'quality_group', 'quantity_group',
        'source_type', 'source_class', 'waterpoint_type_group', 'permit',
        'scheme_name', 'public_meeting'
    ]
    df.drop(columns_to_drop, axis=1, inplace=True)

    #imputting categorical NA
    for col in ['funder', 'installer']:
        df[col] = df[col].fillna("Unknown").replace('0', 'Unknown')

    #Replace not 0 values
    df['amount_tsh'] = df['amount_tsh'].replace(0.0, np.nan)
    df['gps_height'] = df['gps_height'].replace(0, np.nan)
    df['population'] = df['population'].replace(0, np.nan)
    df['longitude'] = df['longitude'].replace(0.0, np.nan)
    df['latitude'] = df['latitude'].replace(0.0, np.nan)
    df['construction_year'] = df['construction_year'].replace(0, np.nan)

    #imputting numerical NA by hierarchy
    df["gps_height"]=df["gps_height"].fillna(df.groupby(['region', 'district_code'])["gps_height"].transform("mean"))
    df["gps_height"]=df["gps_height"].fillna(df.groupby(['region'])["gps_height"].transform("mean"))
    df["gps_height"]=df["gps_height"].fillna(df.groupby(['district_code'])["gps_height"].transform("mean"))
    df["gps_height"]=df["gps_height"].fillna(df["gps_height"].mean())

    df["amount_tsh"]=df["amount_tsh"].fillna(df.groupby(['region', 'district_code'])["amount_tsh"].transform("median"))
    df["amount_tsh"]=df["amount_tsh"].fillna(df.groupby(['region'])["amount_tsh"].transform("median"))
    df["amount_tsh"]=df["amount_tsh"].fillna(df.groupby(['district_code'])["amount_tsh"].transform("median"))
    df["amount_tsh"]=df["amount_tsh"].fillna(df["amount_tsh"].median())

    df["population"]=df["population"].fillna(df.groupby(['region', 'district_code'])["population"].transform("median"))
    df["population"]=df["population"].fillna(df.groupby(['region'])["population"].transform("median"))
    df["population"]=df["population"].fillna(df.groupby(['district_code'])["population"].transform("median"))
    df["population"]=df["population"].fillna(df["population"].median())

    df["longitude"]=df["longitude"].fillna(df.groupby(['region', 'district_code'])["longitude"].transform("mean"))
    df["longitude"]=df["longitude"].fillna(df.groupby(['region'])["longitude"].transform("mean"))

    df["construction_year"] = df["construction_year"].fillna(df.groupby(['region', 'district_code'])["construction_year"].transform("median"))
    df["construction_year"] = df["construction_year"].fillna(df.groupby(['region'])["construction_year"].transform("median"))
    df["construction_year"] = df["construction_year"].fillna(df.groupby(['district_code'])["construction_year"].transform("median"))
    df["construction_year"] = df["construction_year"].fillna(df["construction_year"].median())

    #add operation time
    df['date_recorded'] = pd.to_datetime(df['date_recorded'])
    df['operation_time'] = df['date_recorded'].dt.year - df['construction_year']
    df.drop(['date_recorded','construction_year'], axis=1, inplace=True)

    #formating
    cat_features = df.select_dtypes('object')
    for col in cat_features.columns:
        df[col] = (
            df[col]
            .str.lower()
            .str.strip())

    #group installers and funders by most frequent
    df['top_installer'] = df['installer'].apply(
    lambda x: x if x in df['installer'].value_counts().head(20).index else 'Others')
    df['top_funders'] = df['funder'].apply(
    lambda x: x if x in df['funder'].value_counts().head(20).index else 'Others'
)

    df = df.drop(['funder', 'installer'],axis=1)

    pd.DataFrame(df).to_csv("clean_df.csv")

    return df
