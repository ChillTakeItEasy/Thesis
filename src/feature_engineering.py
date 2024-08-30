from dataclasses import dataclass
import holidays
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
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
        return df.drop(columns=f"lag_{var_to_predict}")
    
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
    @staticmethod
    def compute_optimal_lags(df,config,var_use):
        if not config["ts_into_tab"]["compute_lags"]:
            print(f'Set by hand number of lags: {config["ts_into_tab"]["steps"]}')

            return config["ts_into_tab"]["steps"]
        criterion_values = []

        metric= config["ts_into_tab"]["metric"]
        for lag in range(1, config["ts_into_tab"]["max_lags"] + 1):
            try:
                model = AutoReg(df[var_use], lags=lag).fit()
            except Exception:
                print("df too small")
                print(f"collapse at lag {lag}")
                print("returning prior best")
                try:
                    optimal_lag = np.argmin(criterion_values) + 1
                except Exception:
                    return 1

                
            if metric=="BIC":
                criterion_values.append(model.bic)
            elif metric=="HQIC":
                criterion_values.append(model.hqic)
            else:
                criterion_values.append(model.aic)
        optimal_lag = np.argmin(criterion_values) + 1
        print(f'Optimal number of lags: {optimal_lag}')
        return optimal_lag

@dataclass
class feature_engineering:
    @staticmethod
    def get_mean_variable(df_,variable,n_inicial,n_fin):
        df=df_.copy()
        
        df[f'mean_{variable}_{n_inicial}_{n_fin}']=df.apply(
                    lambda row: feature_engineering.personalize_rolling_mean(df[variable], row.name, int(n_inicial), int(n_fin)), axis=1
                            )
        
        return df   
    
    @staticmethod
    def personalize_rolling_mean(series, current_index, start_back, end_back):
        try:
            start_index = max(0, current_index - start_back)
            end_index = max(0, current_index - end_back )
            return series[end_index:start_index].mean()
        except:
            return np.nan
    
    
    
    @staticmethod
    def get_median_variable(df_,variable,n_inicial,n_fin):
        df=df_.copy()
        
        
        df[f'median_{variable}_{n_inicial}_{n_fin}']=df.apply(
                    lambda row: feature_engineering.personalize_rolling_median(df[variable], row.name, int(n_inicial), int(n_fin)), axis=1
                            )
        
        
        return df   
    
    @staticmethod
    def personalize_rolling_median(series, current_index, start_back, end_back):
        try:
            start_index = max(0, current_index - start_back)
            end_index = max(0, current_index - end_back )
            return series[end_index:start_index].median()
        except:
            return np.nan
    
    
    @staticmethod
    def get_mean_weighthed_variable(df_,variable,n_inicial,n_fin):
        df=df_.copy()
        
        rango=int(n_fin)-int(n_inicial) + 2
        
        if rango > len(df):
            weights = np.arange(1, len(df)+1)
            weights = weights / weights.sum()
        else:
            weights = np.arange(1, rango)
            weights = weights / weights.sum()

        


        df[f'mean_weighted_{variable}_{n_inicial}_{n_fin}']=df.apply(
                    lambda row: feature_engineering.personalize_rolling_weighted_mean(df[variable], row.name, int(n_inicial), int(n_fin)), axis=1
                            )
        
        return df   
    
    @staticmethod
    def personalize_rolling_weighted_mean(series, current_index, start_back, end_back):
        start_index = max(0, current_index - start_back)
        end_index = max(0, current_index - end_back)
        weights = np.arange(1, start_index - end_index+1)
        values = series[end_index:start_index]
        try:
            return np.average(values, weights=weights)
        except:
            return  float('nan')
    @staticmethod    
    def add_time_variable(df_,config):
        df=df_.copy()
        if config["preprocessing"]["aggr_time_level"]=="weekly":
            df['Year'] =  df['date'].dt.year
            df['Week'] =  df['date'].dt.isocalendar().week
            df['Year']=df['Year'].astype(str)
            df['Week']=df['Week'].astype(str)

            df['Week'] = df.apply(lambda x: 0 if (int(x['Week']) >= 52 and x['date'].month == 1) else x['Week'], axis=1)




        elif config["preprocessing"]["aggr_time_level"]=="monthly":
            df["date"]=df["date"].astype(str)
            df[['Year', 'Month']] = df['date'].str.split('-', expand=True).iloc[:, :2]

        elif config["preprocessing"]["aggr_time_level"]=="daily":
            df["date"]=df["date"].astype(str)
            df[['Year', 'Month',"Day"]] = df['date'].str.split('-', expand=True)
            
        return df

    @staticmethod
    def add_holidays_feature(df,country,years_list,time_lvl):
        country_holidays=pd.DataFrame(((date, 1) for date in holidays.country_holidays(country, years=years_list)),columns=["date","holidays"])
        country_holidays["date"]=pd.to_datetime(country_holidays["date"])
        if time_lvl=="weekly":
                df_date = country_holidays["date"]

                # Extract year and week number
                year_week_monday = df_date - pd.to_timedelta(df_date.dt.dayofweek, unit='D')
                
                country_holidays["date"]=year_week_monday
                country_holidays_aggr=country_holidays.groupby("date")["holidays"].sum()
                result = pd.merge(df, country_holidays_aggr, on='date', how='left').fillna(0)
                result["holidays_fe"]=result["holidays"].fillna(0)
                
               
                
                return result
                
        elif time_lvl=="monthly":
                df_date = country_holidays["date"]


                # Extract year and week number
                year_week = df_date.dt.strftime('%Y-%m')
                country_holidays["date"]=year_week
                country_holidays_aggr=country_holidays.groupby("date")["holidays"].sum().reset_index()
                country_holidays_aggr["date"]=pd.to_datetime(country_holidays_aggr["date"])
                
                
                
                result = pd.merge(df, country_holidays_aggr, on='date', how='left').fillna(0)
                result["holidays_fe"]=result["holidays"].fillna(0)
                return result
                
        elif time_lvl=="daily":
                df_date = country_holidays["date"]
                year_week = df_date.dt.strftime('%Y-%m-%d')

                country_holidays["date"]=year_week

                df["date"] = pd.to_datetime(df["date"])
                country_holidays["date"] = pd.to_datetime(country_holidays["date"])

                df["holidays_fe"]=df["date"].isin(country_holidays["date"]).astype(int)

                return df 


        