from dataclasses import dataclass
import pmdarima 
from sktime.forecasting.tbats import TBATS
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt
from prophet import Prophet
import pandas as pd
import numpy as np
import itertools
from prophet.diagnostics import cross_validation,performance_metrics
import logging


@dataclass
class TsModels:
    @staticmethod
    def autoarima(df,df_future,config,use_fuller):
        if use_fuller:
            diff_compute="eda"
        else:
            diff_compute="diff"
        if use_fuller:
            
                model=pmdarima.auto_arima( 
                df[config["var_in_use"]],
                start_p=0,
                start_q=0,
                max_p=config["model"]["arima"]["max_p"],
                max_q=config["model"]["arima"]["max_q"],
                d=0,
                D=None,
                stepwise=False
            )

        else:
            model=pmdarima.auto_arima( 
                df[config["var_in_use"]],
                start_p=0,
                start_q=0,
                max_p=config["model"]["arima"]["max_p"],
                max_q=config["model"]["arima"]["max_q"],
                d=None, 
                D=None,
                stepwise=False #No computation time limit
            ) 

        forecast=model.predict(1)
        

        return [float(forecast.iloc[0]),f"arima_{diff_compute}"]

    @staticmethod
    def autoarima_exogenous(df,df_future,config,use_fuller):
        df=df.dropna()
        if use_fuller:
            diff_compute="eda"
        else:
            diff_compute="diff"

        if use_fuller:
    
                model=pmdarima.auto_arima( 
                df[config["var_in_use"]],
                X=df[config["ts_exo_columns"]],
                start_p=0,
                start_q=0,
                max_p=config["model"]["arima"]["max_p"],
                max_q=config["model"]["arima"]["max_q"],
                d=0,                 D=None,
                stepwise=False #No computation time limit
            )

        else:
            model=pmdarima.auto_arima( 
                df[config["var_in_use"]],
                X=df[config["ts_exo_columns"]],
                start_p=0,
                start_q=0,
                max_p=config["model"]["arima"]["max_p"],
                max_q=config["model"]["arima"]["max_q"],
                d=None,
                D=None,
                stepwise=False #No computation time limit
            )


        
        
        forecast=model.predict(n_periods=1, X=df_future[config["ts_exo_columns"]])
        
        


        return [float(forecast.iloc[0]),f"arima_exo_{diff_compute}"]

    @staticmethod
    def tbats(df,df_future,config):

        if config["model"]["tbats"]["optimize_params"]:

            estimator=TBATS(sp=config["model"]["tbats"]["max_sp"])
            model=estimator.fit( df[config["var_in_use"]]) 

            if config["model"]["tbats"]["optimize_only_once"]:
                config["model"]["tbats"]["optimize_params"]=model.get_params()
                config["model"]["tbats"]["tbats_params"]=False
        else:
            estimator=TBATS(**config["model"]["tbats"]["tbats_params"])
            model=estimator.fit( df[config["var_in_use"]]) 

        forecast=model.predict(fh=1)
            

        forecast=model.predict(fh=len(df_future))

        return [float(forecast.iloc[0]),"tbats"]


    
    @staticmethod
    def exp_smooth(df,df_future,config):

        possibilities_trend=['add','mul']
        possibilities_seasonal=['add','mul']
        possibilities_period=[4,7,15,30]
        Aic=[]
        models=[]
        for trend in possibilities_trend:
            for season in possibilities_seasonal:
                for period in possibilities_period:
                    try:
                        HW = ExponentialSmoothing(df[config["var_in_use"]],
                                                        trend=trend,
                                                        seasonal=season,
                                                        seasonal_periods = period)
                        
                        HW_Model = HW.fit()
                        models.append(HW_Model)
                        Aic.append(HW_Model.aic)
                    except:
                        pass
                        


        best_model=models[Aic.index(min(Aic))]
        
        forecast=best_model.forecast(1)
        
       
        return [float(forecast.iloc[0]),"exp_smooth"]

    
    @staticmethod
    def holt(df,df_future,config):

        HW = Holt(df[config["var_in_use"]])
                        
        model = HW.fit()
        
        
        forecast=model.forecast(1)
        
    
        return [float(forecast.iloc[0]),"holt"]

    @staticmethod
    def prophet(df,df_future,config):

        df_=df.rename(columns={"date":"ds", config["var_in_use"]:"y"})
        df_=df_[["ds","y"]]
        df_["ds"] = pd.to_datetime(df_["ds"])
        
        holiday=pd.DataFrame()

        if "holidays_fe" in df.columns:
            if config["feature_eng"]["add_holidays"]:
                if config["preprocessing"]["aggr_time_level"]=="daily":
                    holiday=df[df["holidays_fe"]==1].copy()
                    holiday.loc[:, "lower_window"] = 0
                    holiday.loc[:, "upper_window"] = 1
                    holiday.loc[:, "holiday"] = "national_holiday"
    
                    holiday=holiday[["date","lower_window","upper_window","holiday"]]
                    holiday=holiday.rename(columns={"date":"ds"})
                    
                    holiday_fut=df[df["holidays_fe"]==1].copy()
                    holiday_fut.loc[:, "lower_window"] = 0
                    holiday_fut.loc[:, "upper_window"] = 1
                    holiday_fut.loc[:, "holiday"] = "national_holiday"
    
                    holiday_fut=holiday_fut[["date","lower_window","upper_window","holiday"]]
                    holiday_fut=holiday_fut.rename(columns={"date":"ds"})
                    
                    holiday = pd.concat([df for df in [holiday, holiday_fut] if not df.empty])


        if config["model"]["prophet"]["optimize_params"]:
            params=TsModels.optimize_proph(df_,config,holiday)
         
            if "holidays_fe" in df.columns:
                if config["feature_eng"]["add_holidays"]:
                    if config["preprocessing"]["aggr_time_level"]=="daily":
                        model =  Prophet(**params,holidays=holiday)
                    else:
                        model=Prophet(**params)
                else:
                    model=Prophet(**params)
            else:
                model =  Prophet(**params)
          

            if config["model"]["prophet"]["optimize_only_once"]:
                config["model"]["prophet"]["optimize_params"]=False
                config["model"]["prophet"]["prophet_params"]=params


        else:
            params=config["model"]["prophet_ex"]["prophet_ex_params"]
            if "holidays_fe" in df.columns:
                if config["feature_eng"]["add_holidays"]:
                    if config["preprocessing"]["aggr_time_level"]=="daily":
                        model =  Prophet(**params,holidays=holiday)
                    else:
                        model=Prophet(**params)
                else:
                    model=Prophet(**params)
            else:
                model =  Prophet(**params)

        try:
            if params["growth"]=="logistic":
                if config["model"]["prophet"]["cap"]:
                    df_["cap"]=config["model"]["prophet"]["cap"]
                    df_future["cap"]=config["model"]["prophet"]["cap"]
                else:
                    df_["cap"]=max(df_["y"])*1.1
                    df_future["cap"]=max(df_["y"])*1.1
        except:
            pass

        
        model.fit(df_)



        
        df_fut=df_future.rename(columns={"date":"ds", config["var_in_use"]:"y"})

        try:
            df_fut=df_fut[["ds","cap"]]
        except:
            df_fut=df_fut[["ds"]]
            
            
        df_fut["ds"] = pd.to_datetime(df_fut["ds"])
        
      
        forecast = model.predict(df_fut)
        
        

        return [float(forecast["yhat"].iloc[0]),"prophet"]
    
    @staticmethod
    def optimize_proph(df,config,holiday):
        logging.disable(logging.CRITICAL)

        

        param_grid=config["model"]["prophet"]["prophet_grid"]
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        rmses = [] 

        
        for params in all_params:
            try:
                if params["growth"]=="logistic":
                   if config["model"]["prophet"]["cap"]:
                       df["cap"]=config["model"]["prophet"]["cap"]
                   else:
                       df["cap"]=max(df["y"])*1.1
            except Exception:
                pass
            

            if "holidays_fe" in df.columns:
                if config["feature_eng"]["add_holidays"]:
                    if config["preprocessing"]["aggr_time_level"]=="daily":
                        m = Prophet(**params,holidays=holiday).fit(df) 
                    else:
                        m = Prophet(**params).fit(df)
                else:
                    m = Prophet(**params).fit(df) 

            else:
                m = Prophet(**params).fit(df) 
            
            if config["model"]["prophet"]["parallel"]:
                df_cv = cross_validation(m,initial=f"{config['model']['prophet']['prophet_initial']} days"  ,horizon=f"{config['model']['prophet']['prophet_horizon']} days",parallel="processes")
            else:
                df_cv = cross_validation(m, initial=f"{config['model']['prophet']['prophet_initial']} days"  ,horizon=f"{config['model']['prophet']['prophet_horizon']} days",disable_tqdm=True)

            df_p = performance_metrics(df_cv, rolling_window=1)
            rmses.append(df_p['rmse'].values[0])

        return all_params[np.argmin(rmses)]


    @staticmethod
    def prophet_exogenous(df,df_future,config):
        exo_columns= config["ts_exo_columns"]
        df=df.dropna()

        df_=df.rename(columns={config["data_extraction"]["ts_column"]:"ds", config["var_in_use"]:"y"})
        df_=df_[["ds","y"]+exo_columns]
        df_["ds"] = pd.to_datetime(df_["ds"])
        
        holiday=pd.DataFrame()
        if "holidays_fe" in df.columns:
           if config["feature_eng"]["add_holidays"]:
               if config["preprocessing"]["aggr_time_level"]=="daily":
                holiday=df[df["holidays_fe"]==1].copy()
                holiday.loc[:, "lower_window"] = 0
                holiday.loc[:, "upper_window"] = 1
                holiday.loc[:, "holiday"] = "national_holiday"

                holiday=holiday[["date","lower_window","upper_window","holiday"]]
                holiday=holiday.rename(columns={"date":"ds"})
                exo_columns.remove("holidays_fe")
                
                holiday_fut=df[df["holidays_fe"]==1].copy()
                holiday_fut.loc[:, "lower_window"] = 0
                holiday_fut.loc[:, "upper_window"] = 1
                holiday_fut.loc[:, "holiday"] = "national_holiday"


                holiday_fut=holiday_fut[["date","lower_window","upper_window","holiday"]]
                holiday_fut=holiday_fut.rename(columns={"date":"ds"})
                
                holiday = pd.concat([df for df in [holiday, holiday_fut] if not df.empty])

  

        if config["model"]["prophet_ex"]["optimize_params"]:
            params=TsModels.optimize_proph(df_,config,holiday,exo_columns)
            
            if "holidays_fe" in df.columns:
                if config["feature_eng"]["add_holidays"]:
                    if config["preprocessing"]["aggr_time_level"]=="daily":
                        model =  Prophet(**params,holidays=holiday)
                    else:
                        model=Prophet(**params)
                else:
                    model=Prophet(**params)
            else:
                model =  Prophet(**params)
            
            if config["model"]["prophet_ex"]["optimize_only_once"]:
                config["model"]["prophet_ex"]["optimize_params"]=False
                config["model"]["prophet_ex"]["prophet_ex_params"]=params
            
                
        else:
            params=config["model"]["prophet_ex"]["prophet_ex_params"]
            if "holidays_fe" in df.columns:
                if config["feature_eng"]["add_holidays"]:
                    if config["preprocessing"]["aggr_time_level"]=="daily":
                        model =  Prophet(**params,holidays=holiday)
                    else:
                        model=Prophet(**params)
                else:
                    model=Prophet(**params)
            else:
                model =  Prophet(**params)
            for regressor in exo_columns:
                model.add_regressor(regressor)

        model.fit(df_)

        try:
            if params["growth"]=="logistic":
                if config["model"]["prophet_ex"]["cap"]:
                    df_["cap"]=config["model"]["prophet_ex"]["cap"]
                    df_future["cap"]=config["model"]["prophet_ex"]["cap"]
                else:
                    df_["cap"]=max(df_["y"])*1.1
                    df_future["cap"]=max(df_["y"])*1.1
        except:
            pass
        model.fit(df_)

    
        df_fut=df_future.rename(columns={config["data_extraction"]["ts_column"]:"ds", config["var_in_use"]:"y"})
        try:
            df_fut=df_fut[["ds","cap"]+config["ts_exo_columns"]]
        except:
            df_fut=df_fut[["ds"]+config["ts_exo_columns"]]
        
        df_fut["ds"] = pd.to_datetime(df_fut["ds"])
        
       
        forecast = model.predict(df_fut)


        return [float(forecast["yhat"].iloc[0]),"prophet_exo"]
    
    @staticmethod
    def optimize_exo_proph(df,config,holiday,exo_columns):
        logging.disable(logging.CRITICAL)


        param_grid=config["model"]["prophet_ex_grid"]
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        rmses = []  # Store the RMSEs for each params here

        for params in all_params:
            try:
                if params["growth"]=="logistic":
                    if params["growth"]=="logistic":
                       if config["model"]["prophet_ex"]["cap"]:
                           df["cap"]=config["model"]["prophet"]["cap"]
                       else:
                           df["cap"]=max(df["y"])*1.1
            except Exception:
                pass
            if "holidays_fe" in df.columns:
                if config["feature_eng"]["add_holidays"]:
                    if config["preprocessing"]["aggr_time_level"]=="daily":
                        m = Prophet(**params,holidays=holiday)
                    else:
                        m = Prophet(**params) 
                else:
                    m = Prophet(**params)
            else:
                m = Prophet(**params)
                
            for regressor in exo_columns:
                m.add_regressor(regressor)
            m.fit(df) 
            if config["model"]["prophet_ex"]["parallel"]:
                df_cv = cross_validation(m, initial=f"{config['model']['prophet_ex']['prophet_initial']} days"  ,horizon=f"{config['model']['prophet_ex']['prophet_horizon']} days",parallel="processes")
            else:
                df_cv = cross_validation(m, iinitial=f"{config['model']['prophet_ex']['prophet_initial']} days"  ,horizon=f"{config['model']['prophet_ex']['prophet_horizon']} days",disable_tqdm=True)
            df_p = performance_metrics(df_cv, rolling_window=1)

            rmses.append(df_p['rmse'].values[0])
            

       
        return all_params[np.argmin(rmses)]

    


    @staticmethod
    def get_best_model(model_list,df_future,config):
        list_errors=[]
        for item in model_list:
            numbers=[item_[0] for item_ in item]
            errors=df_future[config["var_in_use"]]-numbers
            mean_error=np.mean(abs(errors))
            list_errors.append(mean_error)
        
        min_error_index=list_errors.index(min(list_errors))
        best_model_name=model_list[1,min_error_index]
        best_error=min(list_errors)
        print("The best model has been: ",best_model_name)
        print("With an mean absolute error of: ",best_error)
        return best_model_name

    @staticmethod
    def get_average_all_models(model_list,df_future,config):
        list_errors=[]
        for item in model_list:
            numbers=[item_[0] for item_ in item]
            errors=df_future[config["var_in_use"]]-numbers
            mean_error=np.mean(abs(errors))
            list_errors.append(mean_error)
        
        error_mean=np.mean(list_errors)
        
        print("Error of mean of absolute errors: ",error_mean)
        return error_mean
    
