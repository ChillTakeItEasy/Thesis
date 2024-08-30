from dataclasses import dataclass
import pandas as pd
import datetime
import numpy as np
@dataclass
class Preprocessing:
    @staticmethod
    def shift_within_group(group,columns_to_shit):
        group[columns_to_shit] = group[columns_to_shit].shift(periods=1)
        return group
    @staticmethod
    def move_one_back(df_,config):
        df=df_.copy()

        #In order to move ninformation 1 row back as we do not know the future, not necessary when only working with ts data #TODO: ver como cambia si es period=1 o -1, cambiar ascending o descending
        if config["type_analysis"]=="ts_numeric":
            df=df.sort_values(by="date")

            df[config["preprocessing"]["columns_to_shift"]] = df[config["preprocessing"]["columns_to_shift"]].shift(periods=1)


        elif  config["type_analysis"]=="ts_num_cat":
            df=df.sort_values(by=[config["preprocessing"]["aggr_lvl"],"date"])
            df = df.groupby(config["preprocessing"]["aggr_attr"]).apply(Preprocessing.shift_within_group,config["preprocessing"]["columns_to_shift"])

        else:
            df=df.sort_values(by="date")
            
        

        return df
    @staticmethod
    def take_out_na(df,config):
        df=df.dropna(subset=["date"])

        if config["preprocessing"]["deal_na"]=="substitute_by_value":
            

            try:
                df= df.fillna(config["preprocessing"]["substitute_na_value"])
            except:
                df= df.fillna(0)

            df=df.dropna()
            return df.reset_index(drop=True)
        elif config["preprocessing"]["deal_na"]=="substitute_by_mean":
            

            df= df.fillna(df.mean())
            df=df.dropna()
            return df.reset_index(drop=True)
        elif config["preprocessing"]["deal_na"]=="substitute_by_median":
            

            df= df.fillna(df.median())
            df=df.dropna()
            return df.reset_index(drop=True)
        elif config["preprocessing"]["deal_na"]=="substitute_backward":
            
            df = df.fillna(method='ffill')

            return df.reset_index(drop=True)
        elif config["preprocessing"]["deal_na"]=="substitute_forward":
            

            df = df.fillna(method='bfill')

            return df.reset_index(drop=True)
        elif config["preprocessing"]["deal_na"]=="substitute_interpolate":
            

            try:
                if config["preprocessing"]["interpolation_approach"]=="polynomial" | config["preprocessing"]["interpolation_approach"]=="spline":
                    df = df.interpolate(method=config["preprocessing"]["interpolation_approach"],order=config["preprocessing"]["interpolation_order"])
            except:
                df = df.interpolate(method="linear")
            return df
        else:
            df=df.dropna()
            return df.reset_index(drop=True)


    @staticmethod
    def adjust_year_prefix(date_str):
        # Extract the first two characters (which represent the year prefix)
        year = int(date_str[:2])
        month = int(date_str[3:])
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        # Determine the correct century prefix
        if year <= current_year:
            if month < current_month:
                prefix = '20'
            else:
                prefix = '19'
        else:
            prefix = '19'
        
        # Construct the adjusted date string
        adjusted_date_str = f"{prefix}{date_str}"
        
        return adjusted_date_str
    @staticmethod
    def transform_date(df_,config):
        df=df_.copy()
        if config["preprocessing"]["date_type"]=="0Y-0M": 
            df[config["data_extraction"]["ts_column"]] = \
            df[config["data_extraction"]["ts_column"]].apply(Preprocessing.adjust_year_prefix)
            df[config["data_extraction"]["ts_column"]] =  \
                pd.to_datetime(df[config["data_extraction"]["ts_column"]], format='%Y-%m')
                
                
        elif config["preprocessing"]["date_type"]=="Y-M-D H:M:S":
            df["date_subs"] = pd.to_datetime(df[config["data_extraction"]["ts_column"]], errors='coerce')
            
            
            # Extract the date part
            df["date_subs"] = df["date_subs"].dt.date
            
            df.drop(columns=[config["data_extraction"]["ts_column"]],inplace=True)
            
            df=df.rename(columns={"date_subs":config["data_extraction"]["ts_column"]})
            
            
            df[config["data_extraction"]["ts_column"]] =  \
                pd.to_datetime(df[config["data_extraction"]["ts_column"]])
                

                      

        else:
            df[config["data_extraction"]["ts_column"]] = \
                  pd.to_datetime(df[config["data_extraction"]["ts_column"]])

        return df.sort_values(config["data_extraction"]["ts_column"])



    #TODO: ADD FECHAS BIEN CON TODO LOS DATOS 
    @staticmethod
    def get_time_groups(df_,config):
        
        df=df_.copy()
        
        if config["preprocessing"]["aggr_time"]:

            if config["preprocessing"]["aggr_time_level"]=="weekly":
                df_date = df[config["data_extraction"]["ts_column"]]


                # Calculate the Monday of the week for each date
                monday = df_date - pd.to_timedelta(df_date.dt.dayofweek, unit='D')

                # Extract year and week number
                df["date"]=monday

            elif config["preprocessing"]["aggr_time_level"]=="monthly":
                df_date = df[config["data_extraction"]["ts_column"]]


                # Extract year and week number
                year_week = df_date.dt.strftime('%Y-%m')
                
                df["date"]=pd.to_datetime(year_week, format='%Y-%m')
            elif config["preprocessing"]["aggr_time_level"]=="daily":
                df_date = df[config["data_extraction"]["ts_column"]]
                year_week = df_date.dt.strftime('%Y-%m-%d')

                df["date"]=pd.to_datetime(year_week, format='%Y-%m-%d')
                

        else:
            df["date"]=df[config["data_extraction"]["ts_column"]]
        return df
    
    @staticmethod
    def fill_empty_time(df,config):
        min_date = df['date'].min()
        max_date = df['date'].max()
        if config["preprocessing"]["aggr_time_level"] == 'daily':
            full_range = pd.date_range(start=min_date, end=max_date, freq='D')
        elif config["preprocessing"]["aggr_time_level"] == 'weekly':
            full_range = pd.date_range(start=min_date, end=max_date, freq='W-MON')
        elif config["preprocessing"]["aggr_time_level"] == 'monthly':
            full_range = pd.date_range(start=min_date, end=max_date, freq='MS')

        full_range_df = pd.DataFrame(full_range, columns=['date'])
        df_full = pd.merge(full_range_df, df, on='date', how='left')


        return df_full
    
    @staticmethod
    def group_by_time(df_,config):
        df=df_.copy()

        if config["type_analysis"]=="ts_num_cat":
            means=df.groupby([config["preprocessing"]["aggr_attr"],"date"])[config["preprocessing"]["mean_columns"]].mean().reset_index()
            sums=df.groupby([config["preprocessing"]["aggr_attr"],"date"])[config["preprocessing"]["sum_columns"]].sum().reset_index()
            cat_columns=df.groupby([config["preprocessing"]["aggr_attr"],"date"])[config["preprocessing"]["unique"]].first().reset_index() 

            df_fin=means.merge(sums,how="inner",on=[config["preprocessing"]["aggr_attr"],"date"])
            df_fin=df_fin.merge(cat_columns,how="inner",on=[config["preprocessing"]["aggr_attr"],"date"])


        elif config["type_analysis"]=="ts_numeric":
            means=df.groupby("date")[config["preprocessing"]["mean_columns"]].mean().reset_index()
            sums=df.groupby("date")[config["preprocessing"]["sum_columns"]].sum().reset_index()
            df_fin=means.merge(sums,how="inner",on="date")
        else:
            if config["preprocessing"]["sum_columns"]:
                df_fin=df.groupby("date")[config["preprocessing"]["sum_columns"]].sum().reset_index()
                
            else:
                df_fin=df.groupby("date")[config["preprocessing"]["mean_columns"]].mean().reset_index()



        
        
        if config["preprocessing"]["cut_extremes"]:
            return Preprocessing.drop_first_row_by_grp(Preprocessing.drop_last_row_by_grp(df_fin,config),config)
        else: 
            return df_fin
    @staticmethod    
    def divide_specific_columns_by_time(df_,config):
        df=df_.copy()

        df[config["preprocessing"]["columns_to_divide_by_time"]]=df[config["preprocessing"]["columns_to_divide_by_time"]].div(config["preprocessing"]["number_to_divide_by_time"])

        return df

    

    @staticmethod        
    def drop_first_row_by_grp(df_,config):
        df=df_.copy()
        if config["type_analysis"]=="ts_num_cat":
            df = df.groupby(config["preprocessing"]["aggr_lvl"], as_index=False).apply(Preprocessing.drop_first_of_each_category).reset_index(drop=True)

        else:
            df.drop(df.index[0], inplace=True)
            
        return df
    @staticmethod        
    def drop_last_row_by_grp(df_,config):
        df=df_.copy()
        if config["type_analysis"]=="ts_num_cat":
            df = df.groupby(config["preprocessing"]["aggr_lvl"], as_index=False).apply(Preprocessing.drop_last_of_each_category).reset_index(drop=True)

        else:
            df.drop(df.index[-1], inplace=True)
            
        return df

    @staticmethod
    def drop_first_of_each_category(group):
        return group.iloc[1:]
    
    @staticmethod
    def drop_last_of_each_category(group):
        return group.iloc[:-1]
    @staticmethod
    def min_max_scaler(X):
        min_values = X.min(axis=0)
        max_values = X.max(axis=0)
        X_normalized = (X - min_values) / (max_values - min_values)
        return (X_normalized, min_values, max_values)
    @staticmethod
    def de_min_max_scaler(X_normalized, min_values, max_values):
        X_denormalized = X_normalized * (max_values - min_values) + min_values
        return X_denormalized
    @staticmethod
    def standardize(data):
        mean = np.mean(data)
        std = np.std(data)
        standardized_data = (data - mean) / std
        return (standardized_data, mean, std)
    @staticmethod
    def destandardize(standardized_data, mean, std):
        original_data = standardized_data * std + mean
        return original_data
    @staticmethod
    def robust_scale(data):
        median = np.median(data)
        q25, q75 = np.percentile(data, [25, 75])
        iqr = q75 - q25
        scaled_data = (data - median) / iqr
        return (scaled_data, median, iqr)
    @staticmethod
    def robust_denormalize(scaled_data, median, iqr):
        original_data = scaled_data * iqr + median
        return original_data



@dataclass
class TsIntoTabular:
    @staticmethod
    def get_difference(df_,config,var_to_predict,diff_name):
        df=df_.copy()
        if config["type_analysis"]=="ts_num_cat":
            try:
                var_to_grp=config["preprocessing"]["aggr_lvl"]
                df[f"lag_{var_to_predict}"]=df.groupby(var_to_grp)[var_to_predict].shift(1).fillna(np.nan)
            except:
                df[f"lag_{var_to_predict}"]=df[var_to_predict].shift(1).fillna(np.nan)

        else:
            df[f"lag_{var_to_predict}"]=df[var_to_predict].shift(1).fillna(np.nan)

        df[diff_name]=df_[var_to_predict]-df[f"lag_{var_to_predict}"]
        return df
    
    @staticmethod
    def get_lag(df_,config,var_use):#TODO: Quitar ifs
        df=df_.copy()

        n=config["ts_into_tab"]["steps"]

        if config["type_analysis"]=="ts_num_cat":
            try:
                var_to_grp=config["preprocessing"]["aggr_attr"]
                df_grouped=df.groupby([var_to_grp])[var_use]
                for i in range(1,n):
                    df[f"lag_{var_use}_"+str(i)]=df_grouped.shift(1).astype(float)
            except:
                df[f"lag_{var_use}_"+str(1)]=df[var_use].shift(1).astype(float)
                for i in range(1,n+1):
                    
                    df[f"lag_{var_use}_"+str(i+1)]=df[f"lag_{var_use}_"+str(i)].shift(1).astype(float)
                    
        else:
            df[f"lag_{var_use}_"+str(1)]=df[var_use].shift(1).astype(float)
            for i in range(1,n):
                df[f"lag_{var_use}_"+str(i+1)]=df[f"lag_{var_use}_"+str(i)].shift(1).astype(float)

        return df