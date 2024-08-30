
import pandas as pd
from dataclasses import dataclass
import os
from pyspark.sql import SparkSession
@dataclass
class ExtractData:
    config:dict
    @staticmethod
    def extract_df_csv(config):
        df=pd.read_csv(fr"{config['ruta_padre']}/assets/data/{config['data_extraction']['name']}.csv")
        return df
    @staticmethod
    def extract_df_parquet(config):
        df=pd.read_parquet(fr"{config['ruta_padre']}/assets/data/{config['data_extraction']['name']}.parquet")
        return df
    @staticmethod
    def extract_df_excel(config):
        if config["data_extraction"]["excel_split"]:
            df_or=pd.read_excel(fr"{config['ruta_padre']}/assets/data/{config['data_extraction']['name']}.xlsx",header=None)
            df = df_or[0].str.split(',', expand=True)
            headers = df.iloc[0]
            df = df[1:]
            df.columns = headers
            df.reset_index(drop=True, inplace=True)

        else:
            df=pd.read_excel(fr"{config['ruta_padre']}/assets/data/{config['data_extraction']['name']}.xlsx")

        return df
    
    
    @staticmethod
    def extract_df_s3_parquet(config):
        spark=SparkSession.builder \
                    .appName("Read Parquet from S3") \
                    .config("spark.hadoop.fs.s3a.access.key", config['data_extraction']['s3_id']) \
                    .config("spark.hadoop.fs.s3a.secret.key", config['data_extraction']['s3_key']) \
                    .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com") \
                    .getOrCreate()
        df=spark.read.parquet(fr"{config['ruta_padre']}/assets/data/{config['data_extraction']['s3_route']}.parquet")
        return df
    @staticmethod
    def extract_df_s3_excel(config):
        spark=SparkSession.builder \
                    .appName("Read Parquet from S3") \
                    .config("spark.hadoop.fs.s3a.access.key", config['data_extraction']['s3_id']) \
                    .config("spark.hadoop.fs.s3a.secret.key", config['data_extraction']['s3_key']) \
                    .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com") \
                    .getOrCreate()
        df=spark.read.parquet(fr"{config['ruta_padre']}/assets/data/{config['data_extraction']['s3_route']}.xlsx")
        return df
    @staticmethod
    def extract_df_s3_csv(config):
        spark=spark = SparkSession.builder \
                    .appName("Read Parquet from S3") \
                    .config("spark.hadoop.fs.s3a.access.key", config['data_extraction']['s3_id']) \
                    .config("spark.hadoop.fs.s3a.secret.key", config['data_extraction']['s3_key']) \
                    .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com") \
                    .getOrCreate()
        df=spark.read.parquet(fr"{config['ruta_padre']}/assets/data/{config['data_extraction']['s3_route']}.csv")
        return df
    @staticmethod    
    def get_needed_info_by_type(df,config):
        def safe_to_numeric(series):
            try:
                return pd.to_numeric(series)
            except ValueError:
                return series  
        df_=df.copy()
        if config["type_analysis"]=="ts_only":
            columns_needed=[config["data_extraction"]["ts_column"]]+[config["data_extraction"]["prediction_column"]]
            df_numeric = df_.apply(safe_to_numeric)

        elif config["type_analysis"]=="ts_numeric":
            
            
            df_numeric = df_.apply(safe_to_numeric)

            numeric_columns = list(df_numeric.select_dtypes(include=['number']).columns)
            date_column=[config["data_extraction"]["ts_column"]]
            essential_column=[config["data_extraction"]["prediction_column"]]
            try:
                numeric_columns.remove(config["data_extraction"]["prediction_column"])
            except:
                pass
            
            for element in config["data_extraction"]["categoric_numeric_columns"]:
                try:
                    numeric_columns.remove(element)
                except:
                    pass

            columns_needed=date_column+essential_column+numeric_columns

            #TODO: añadir lags, ver como se estructuran
        else:
            columns_needed=df.columns()
            df_numeric = df_.apply(safe_to_numeric)

            
        df_essential_info=df_numeric[columns_needed]
        return df_essential_info
    
    
    @staticmethod
    def clean_specific_vars_writing_issues(df,config):
        for problem in config["data_extraction"]["variable_issues"]:
            var=problem[0]
            issue=problem[1]
            
            if issue=="european_format":
                df[var] = df[var].str.replace('.', '', regex=False)  
                df[var] = df[var].str.replace(',', '.', regex=False)  
                df[var] = df[var].astype(float)  
                
            if issue=="pct_european":
                df[var] = df[var].str.replace('%', '', regex=False)  
                df[var] = df[var].str.replace(',', '.', regex=False)  
                df[var] = df[var].astype(float) / 100  


        return df
    

    @staticmethod
    def clean_duplicated_columns_rows(df,config):
        if config["data_extraction"]["clean_columns_rows"]:
            
            df_cleaned = df[~(df[config["data_extraction"]["prediction_column"]]==config["data_extraction"]["prediction_column"])]
            return df_cleaned
        else:
            return df
    
