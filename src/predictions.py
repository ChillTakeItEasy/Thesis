from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import ConvergenceWarning

@dataclass
class SetFunctions:

    @staticmethod
    def get_future_and_past(df,i):
        
            
            # Extract the last 10% of the DataFrame
            past_df = df.iloc[:len(df)-1-i]

            # Extract one more row of the DataFrame
            future_df = df.iloc[[len(df)-1-i]]
            return past_df, future_df.copy()

        
    
    @staticmethod
    def get_all_preds_ts(df,config):
        from .ts_models import TsModels
        rows_to_pred=int(config["model"]["percentage_predicted"])
        all_models=[]

        

        for i in range(rows_to_pred):
            print(i)
            df_past,df_future=SetFunctions.get_future_and_past(df,i)
            
            df_past = df_past.loc[:, df_past.apply(lambda col: col.nunique() > 1, axis=0)]

            columns_=df_past.columns
            not_wanted_columns=["date",config["data_extraction"]["prediction_column"],"difference","secdifference"]
            columns_=list(set(columns_)-set(not_wanted_columns))
            
            config["ts_exo_columns"]=columns_
            
            df_past[columns_]=df_past[columns_].astype(float)

            arima=TsModels.autoarima(df_past,df_future,config,False)
            arima_eda=TsModels.autoarima(df_past,df_future,config,True)
            tbats=TsModels.tbats(df_past,df_future,config)
            exp_smooth=TsModels.exp_smooth(df_past,df_future,config)
            holt=TsModels.holt(df_past,df_future,config)
            prophet=TsModels.prophet(df_past,df_future,config)
            
            try:
                ts_models_list=[arima,arima_eda,tbats,exp_smooth,holt,prophet]
            except:
                ts_models_list=[arima,arima_eda,tbats,exp_smooth,holt]


            if config["type_analysis"]!="ts_only":
                arima_exogenous=TsModels.autoarima_exogenous(df_past,df_future,config,False)
                arima_exogenous_eda=TsModels.autoarima_exogenous(df_past,df_future,config,True)
                try:
                    prophet_exogenous=TsModels.prophet_exogenous(df_past,df_future,config)
                    ts_models_list=ts_models_list+[arima_exogenous,arima_exogenous_eda,prophet_exogenous]
                except:
                    ts_models_list=ts_models_list+[arima_exogenous]
            all_models.append(ts_models_list)

        models_df=SetFunctions.get_df_from_or_list(all_models)
        return models_df
    


    @staticmethod
    def get_all_preds_tab(df,config):
        from .tabular_models import TbModels

        rows_to_pred=int(config["model"]["percentage_predicted"])
        all_models=[]
    
        


        for i in range(rows_to_pred):
            print(i)

            df_past,df_future=SetFunctions.get_future_and_past(df,i)
            df_past = df_past.loc[:, df_past.apply(lambda col: col.nunique() > 1, axis=0)]

            columns_=df_past.columns
            not_wanted_columns=["date",config["data_extraction"]["prediction_column"],"difference","secdifference"]
            columns_=list(set(columns_)-set(not_wanted_columns))
            
            config["tab_exo_columns"]=columns_
            
            df_past[columns_]=df_past[columns_].astype(float)
            df_future[columns_]=df_future[columns_].astype(float)
            
            
            random_forest=TbModels.random_forest(df_past,df_future,config)
            xgboost=TbModels.xgboost(df_past,df_future,config)

            df_past=df_past.dropna(axis=0)

            
            linear_regress=TbModels.linear_regress(df_past,df_future,config)
            svm=TbModels.svm(df_past,df_future,config)
            lasso_regress=TbModels.lasso_regress(df_past,df_future,config)
            ridge_regress=TbModels.ridge_regress(df_past,df_future,config)
            elastic_regress=TbModels.elastic_regress(df_past,df_future,config)
            knn=TbModels.knn(df_past,df_future,config)
            
            tab_models_list=[linear_regress,random_forest,svm,lasso_regress,ridge_regress,elastic_regress,xgboost,knn]
            all_models.append(tab_models_list)

        models_df=SetFunctions.get_df_from_or_list(all_models)
        return models_df
    

    
    @staticmethod
    def get_inverse_diffs(df,df_preds,config):

        return
    @staticmethod
    def get_bagging_models(df_,ts_columns,tab_columns,baseline_columns):
        df=df_.copy()
        approaches=["ts_avg","tab_avg","bsln_avg"]
        approaches_models=[ts_columns,tab_columns,baseline_columns]
        
        
        for i in range(len(approaches)):
            df[approaches[i]]=SetFunctions.get_bagging_prediction(df[approaches_models[i]])

        return df[approaches]
    
    @staticmethod
    def get_bagging_prediction(df):

        return df.mean(axis=1)
    
    @staticmethod
    def get_weighted_bagging_models(df_,ts_columns,tab_columns,baseline_columns,df_real,config):
        df=df_.copy()
        approaches=["bsln_wavg","ts_wavg","tab_wavg"]
        approaches_models=[baseline_columns,ts_columns,tab_columns]
        
        
        for i in range(len(approaches)):
           df[approaches[i]]=SetFunctions.get_weighted_bagging_prediction(df[approaches_models[i]],df_real,config)

        return df[approaches]
    
    @staticmethod
    def get_weighted_bagging_prediction(df,df_real,config):
        len_preds,number_preds=df.shape
        df_real=df_real.tail(len_preds).reset_index(drop=True)

        preds=list(SetFunctions.get_bagging_prediction(df[:number_preds+int(len_preds*0.15)]))
        

        for step in range(number_preds+int(len_preds*0.15),len_preds):

            X = df[:step]
            y = df_real[config["data_extraction"]["prediction_column"]][:step]           

            # Initialize and fit the model
            model = LinearRegression()
            model=model.fit(X, y)
            pred_step=model.predict(df.iloc[[step]])[0]
            preds.append(pred_step)
        return preds



    @staticmethod
    def get_all_preds_baselines(df,config):
        if config["model"]["use_original_for_baseline"]:
            var_save=config["var_in_use"]
            config["var_in_use"]=config["data_extraction"]["prediction_column"]
            
        rows_to_pred=int(config["model"]["percentage_predicted"])
        all_models=[]
        for i in range(rows_to_pred):
            df_past,df_future=SetFunctions.get_future_and_past(df,i)
            last_observation=SetFunctions.last_observation(df_past,config)
            mean_last_observations=SetFunctions.mean_last_observations(df_past,config)
            weighted_mean_last_observations=SetFunctions.weighted_mean_last_observations(df_past,config)
            mean_process=SetFunctions.mean_process(df_past,config)
            median_process=SetFunctions.median_process(df_past,config)
            median_last_observations=SetFunctions.median_last_observations(df_past,config)
            
            tab_models_list=[last_observation,mean_last_observations,mean_process,weighted_mean_last_observations,median_last_observations,median_process]
            all_models.append(tab_models_list)

        models_df=SetFunctions.get_df_from_or_list(all_models)
        if config["model"]["use_original_for_baseline"]:
            config["var_in_use"]=var_save
        return models_df
    
    @staticmethod
    def last_observation(df,config):

        forecast=df[config["var_in_use"]].iloc[-1]


        return [forecast,"last"]
    @staticmethod
    def mean_last_observations(df,config):
        
        forecast=df[config["var_in_use"]].tail(config["ts_into_tab"]["steps"]).mean()
        return [forecast,"mean_latest"]
    
    @staticmethod
    def median_last_observations(df,config):
        
        forecast=df[config["var_in_use"]].tail(config["ts_into_tab"]["steps"]).median()
        return [forecast,"median_latest"]
    
    @staticmethod
    def weighted_mean_last_observations(df, config):
        # Retrieve the column name and the number of steps from the config
        prediction_column = config["var_in_use"]
        steps = config["ts_into_tab"]["steps"]
    
        weights = np.arange(1, steps + 1)
        weights = weights / weights.sum()  # Normalize the weights
        
        # Calculate the weighted mean
        weighted_mean =   df[prediction_column][-steps:].multiply(weights).sum()


        # Return the forecast and identifier
        return [weighted_mean, "weight_last"]
    @staticmethod
    def mean_process(df,config):
        forecast=df[config["var_in_use"]].mean()
        return [forecast,"mean_all"]
    @staticmethod
    def median_process(df,config):
        forecast=df[config["var_in_use"]].median()
        return [forecast,"median"]
    



    @staticmethod
    def get_df_from_or_list(lista):
        
        lista_agrupada = {}

        for sub in lista:
            for sublista in sub:
                
                # Obtener el tercer elemento de la sublista (el string)
                clave = sublista[1]
                # Verificar si la clave ya existe en el diccionario
                if clave in lista_agrupada:
                    # Si la clave existe, agregar la sublista a la lista correspondiente
                    lista_agrupada[clave].append(sublista[0])
                else:
                    # Si la clave no existe, crear una nueva lista con la sublista
                    lista_agrupada[clave] = [sublista[0]]
                    
       

        return pd.DataFrame(lista_agrupada)[::-1].reset_index(drop=True)

