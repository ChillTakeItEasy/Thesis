import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error, median_absolute_error
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import seaborn as sns
import xlsxwriter
class ResultAnalysis:

    def __init__(self,df,config) -> None:
        self.df=df
        self.config=config

    
    def get_metrics(self):
        numeric_columns=self.df.select_dtypes(['number']).columns
        df_metrics=pd.DataFrame()
        df_metrics["Metric"]=["Rms","Mae","Mdn","Mp","Hbb","Q25","Q75","Me"]
        for column in numeric_columns:
            if column!=self.config["data_extraction"]["prediction_column"]:
                y_true=self.df[self.config["data_extraction"]["prediction_column"]]
                y_pred=self.df[column]
                rmse=self.get_rmse(y_true,y_pred)
                mae=self.get_mae(y_true,y_pred)
                me=self.get_me(y_true,y_pred)
                mape=self.get_mape(y_true,y_pred)
               
    
                hubber=self.get_hubber_loss(y_true,y_pred)
                quantile1=self.get_quantile_error(y_true,y_pred,0.25)
                quantile2=self.get_quantile_error(y_true,y_pred,0.75)
                median=self.get_median_error(y_true,y_pred)
                df_metrics[column]=[rmse,mae,median,mape,hubber,quantile1,quantile2,me]



        return df_metrics
    

    
    def get_rmse(self,y_true,y_pred):
        

        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    
    def get_mae(self,y_true,y_pred):
        return mean_absolute_error(y_true, y_pred)
    
    def get_me(self,y_true,y_pred):
        return (y_pred- y_true).mean()
    
    
    def get_mape(self,y_true,y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    
    def get_R_sqrt(self,y_true,y_pred,tabular):
        n= len(y_true)
        if tabular:
            k=self.config["ts_into_tab"]["steps"]*(1+len(self.config["ts_into_tab"]["vars_to_lag"]))+len(self.config["feature_eng"]["mean_vars"])+len(self.config["model"]["exogenous_var"])
        else:
            k=1
        r2 = r2_score(y_true, y_pred)
        return 1 - (1 - r2) * (n - 1) / (n - k - 1)
        

    
    def get_hubber_loss(self,y_true,y_pred,delta=1.0):
        error = y_true - y_pred
        is_small_error = np.abs(error) <= delta
        squared_loss = np.square(error) / 2
        linear_loss = delta * (np.abs(error) - delta / 2)
        return np.mean(np.where(is_small_error, squared_loss, linear_loss))
    

    
    def get_median_error(self,y_true,y_pred):
        return median_absolute_error(y_true, y_pred)

    

    
    def get_quantile_error(self,y_true,y_pred,q=0.5):
        error = y_true - y_pred
        return np.mean(np.maximum(q * error, (q - 1) * error))
    
    @staticmethod
    def show_df(df):

        error_names = ["Rmse", "Mae", "Median", "Mape", "Hubber", "Quantile_025", "Quantile_075", "Me"]
        
        

        df_split = [df.iloc[[i]] for i in range(0, len(df))]
        
        # Iterate through the split DataFrames and plot them
        for i, df_part in enumerate(df_split):
            
            
            df_part=pd.DataFrame(df_part)
            
            df_part=df_part.drop(columns=["Metric"])
            
            df_long = pd.melt(df_part, var_name='Error Model', value_name='Error Value')


            # Update the 'Category' column with the corresponding error names
            
            # Set the title for the plot
            title = error_names[i]
            colors = sns.color_palette("husl", len(df_long['Error Model'].unique()))

            # Plot the current part of the DataFrame
            ax = sns.barplot(x='Error Model', y='Error Value', data=df_long, hue='Error Model', palette=colors, legend=False)


            ax.set_title(title, fontsize=16)

            
            ax.set_xticks(range(len(ax.get_xticklabels())))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')

            # Clean up the plot
            ax.set_xlabel('Error Model')
            ax.set_ylabel('Error amount')
            plt.tight_layout()
            
            # Show the plot
            plt.show()
            
            
    @staticmethod
    def get_conclusions(df,df_metrics,baseline_columns,tab_columns,ts_columns):
        df_baseline_better=df[df["model_selector_name"].isin(baseline_columns)]        

        print()
        print()
        print()
        print(f"Temporal models and tabulars were selected on {100*(1-(len(df_baseline_better)/len(df)))}% of the cases")
        baseline_columns=baseline_columns+["bsln_avg","bsln_wavg"]
        tab_columns=tab_columns+["tab_avg","tab_wavg"]
        ts_columns=ts_columns+["ts_avg","ts_wavg"]

        lowest_baseline=df_metrics[baseline_columns][df_metrics["Metric"]=="Mae"].min(axis=1).mode()[0]
        lowest_baseline_model=df_metrics[baseline_columns][df_metrics["Metric"]=="Mae"].idxmin(axis=1).mode()[0]
        
        lowest_ts=df_metrics[ts_columns][df_metrics["Metric"]=="Mae"].min(axis=1).mode()[0]
        lowest_ts_model=df_metrics[ts_columns][df_metrics["Metric"]=="Mae"].idxmin(axis=1).mode()[0]
        
        lowest_tab=df_metrics[tab_columns][df_metrics["Metric"]=="Mae"].min(axis=1).mode()[0]
        lowest_tab_model=df_metrics[tab_columns][df_metrics["Metric"]=="Mae"].idxmin(axis=1).mode()[0]
        
        selector_mae=df_metrics[["mdl_slct_pred"]][df_metrics["Metric"]=="Mae"].min(axis=1).mode()[0]
        
        print(f"Best baseline model is {lowest_baseline_model} with a MAE of {lowest_baseline}")
        print(f"Best tabular model is {lowest_tab_model} with a MAE of {lowest_tab}")
        print(f"Best time series model is {lowest_ts_model} with a MAE of {lowest_ts}")
        print(f"Model selector has a MAE of {selector_mae}")
        
              
        
        baseline_info=["Best baseline model",lowest_baseline_model,lowest_baseline]
        ts_info=["Best time series model",lowest_ts_model,lowest_ts]
        tab_info=["Best tabular model",lowest_tab_model,lowest_tab]
        selector_info=["Specifc model","mdl_slct_pred",selector_mae]
        
     
        columnas=["Type of approach","Model","MAE"]
        
        df_mae=pd.concat([pd.DataFrame([baseline_info], columns=columnas),pd.DataFrame([ts_info], columns=columnas),pd.DataFrame([tab_info], columns=columnas),pd.DataFrame([selector_info], columns=columnas)])
        
        lowest_baseline=df_metrics[baseline_columns][df_metrics["Metric"]=="Mp"].min(axis=1).mode()[0]
        lowest_baseline_model=df_metrics[baseline_columns][df_metrics["Metric"]=="Mp"].idxmin(axis=1).mode()[0]
        
        lowest_ts=df_metrics[ts_columns][df_metrics["Metric"]=="Mp"].min(axis=1).mode()[0]
        lowest_ts_model=df_metrics[ts_columns][df_metrics["Metric"]=="Mp"].idxmin(axis=1).mode()[0]
        
        lowest_tab=df_metrics[tab_columns][df_metrics["Metric"]=="Mp"].min(axis=1).mode()[0]
        lowest_tab_model=df_metrics[tab_columns][df_metrics["Metric"]=="Mp"].idxmin(axis=1).mode()[0]
        
        selector_mae=df_metrics[["mdl_slct_pred"]][df_metrics["Metric"]=="Mp"].min(axis=1).mode()[0]
        print()
        print()
        print(f"Best percentual baseline model is {lowest_baseline_model} with a Mape of {lowest_baseline}")
        print(f"Best percentual tabular model is {lowest_tab_model} with a Mape of {lowest_tab}")
        print(f"Best percentual time series model is {lowest_ts_model} with a Mape of {lowest_ts}")
        print(f"Model selector has a Mape of {selector_mae}")
        
              
        
        baseline_info=["Best baseline model",lowest_baseline_model,lowest_baseline]
        ts_info=["Best time series model",lowest_ts_model,lowest_ts]
        tab_info=["Best tabular model",lowest_tab_model,lowest_tab]
        selector_info=["Specifc model","mdl_slct_pred",selector_mae]
        
        df_mape=pd.concat([pd.DataFrame([baseline_info], columns=columnas),pd.DataFrame([ts_info], columns=columnas),pd.DataFrame([tab_info], columns=columnas),pd.DataFrame([selector_info], columns=columnas)])

        return pd.concat([df_mae,df_mape])
    @staticmethod
    def save_df(metrics_df,fast_summary,raw,config):
        if config["preprocessing"]["normalize"]:
            path=fr"{config['ruta_padre']}/assets/analysis/{config['data_extraction']['name']}_analysis_{config['preprocessing']['type_normalization']}_{config['var_in_use']}_{config['model']['percentage_predicted']}.xlsx"
        elif config["preprocessing"]["normalize_if_not_diff"]:
            if config["var_in_use"]==config["data_extraction"]["prediction_column"]:
                path=fr"{config['ruta_padre']}/assets/analysis/{config['data_extraction']['name']}_analysis_{config['preprocessing']['type_normalization']}_{config['var_in_use']}_{config['model']['percentage_predicted']}.xlsx"
            else:
                path=fr"{config['ruta_padre']}/assets/analysis/{config['data_extraction']['name']}_analysis_{config['var_in_use']}_{config['model']['percentage_predicted']}.xlsx"
        else:
            path=fr"{config['ruta_padre']}/assets/analysis/{config['data_extraction']['name']}_analysis_{config['var_in_use']}_{config['model']['percentage_predicted']}.xlsx"

        with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
            # Write each DataFrame to a different worksheet.
            metrics_df.to_excel(writer, sheet_name='metrics_df', index=False)
            fast_summary.to_excel(writer, sheet_name='fast_summary', index=False)
            raw.to_excel(writer, sheet_name='raw_data', index=False)
        
        

