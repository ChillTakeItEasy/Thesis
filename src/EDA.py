from dataclasses import dataclass
import matplotlib.pyplot as plt
import logging
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf 
from statsmodels.tsa.seasonal import seasonal_decompose
import time
import pandas as pd
@dataclass
class EDA:

    @staticmethod    
    def plot_evolutions_of_diffs(df,config):
        plt.figure(figsize=(22,10))

        try:
            plt.plot(df["date"],df["difference"],label="First Difference")
        
            try:
                plt.plot(df["date"],df["secdifference"],label="Second Difference")
            except:
                pass
            
            plt.title(f"Time series: {config['data_extraction']['name']}")
            plt.xlabel("Date")
            plt.ylabel("Evolutions")
            plt.show()
        except:
            pass
        
    @staticmethod
    def variable_heatmap(df,config,lag_mode):
        df_to_plot=df.copy()
        if lag_mode:
            lag_columns=[config["var_in_use"]]+ [col for col in df.columns if 'lag' in col ]
            
            
            
            
        else:
            lag_columns=[config["var_in_use"]]
            try:
                lag_columns=lag_columns+config["EDA"]["vars_to_add_heatmap"]
            except:
                pass
            if config["EDA"]["feature_eng_vars"]:
                try:
                    lag_columns= lag_columns+list(df.filter(regex=r'mean_').columns)
                except:
                    pass
                try:
                    lag_columns= lag_columns+list(df.filter(regex=r'median_').columns)
                except:
                    pass

                if config["feature_eng"]["add_holidays"]:
                        lag_columns= lag_columns+["holidays_fe"]

                if config["feature_eng"]["add_date"]:
                    if config["preprocessing"]["aggr_time_level"]=="monthly":
                        lag_columns= lag_columns+["Year","Month"]
        
                    elif config["preprocessing"]["aggr_time_level"]=="weekly":
                        lag_columns= lag_columns+["Year","Week"]
        
                    else:
                        lag_columns= lag_columns+["Year","Month","Day"]
                        
        if len(lag_columns)==1:
            print("NO Correlations to plot")
            return
            
        df_corr=df_to_plot[lag_columns].corr()
        df_corr.style.background_gradient(cmap='coolwarm').set_properties(**{'font-size': '0pt'})

        rows = min(10, df.shape[0])
        columns = min(10, df.shape[1])
        
        # Print the first 10x10 (or less) elements
        print(df.iloc[:rows, :columns])
        
        EDA.plot_heatmap(df_to_plot[lag_columns])
    
        
    @staticmethod
    def plot_heatmap(df):
        df = df.astype(float)
        df = df.loc[:, df.apply(lambda col: col.nunique() > 1, axis=0)]

        f = plt.figure(figsize=(24, 18))
        plt.matshow(df.corr(), fignum=f.number)
        plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=16, rotation=45)
        plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=16)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=12)
        plt.title('Correlation Matrix', fontsize=20)
        plt.show()

    @staticmethod
    def check_stat_adfuller(df_,var_name):
        try:
            df=df_.copy()
            df.dropna(subset=[var_name],inplace=True)
            result = adfuller(df[var_name], autolag='AIC')
            logging.debug('Test statistic: ' , result[0])
            logging.debug('p-value: '  ,result[1])
            logging.debug('Critical Values:' ,result[4])
            print(result[1])
    
            EDA.plot_MA(df,var_name)
    
            return result[1]
        except Exception:
            print("TOO SMALL OF SAMPLE SIZE")
            time.sleep(5)
            return 1
    
    @staticmethod
    def decompose_ts(df,var_to_plot,freq):
        try:
            df_=df[["date",var_to_plot]]
            df_.index = pd.to_datetime(df_['date'])
            df_=df_.dropna()
            if freq=="monthly":
                try:
                    df_.index.freq = 'MS'  # Manually setting frequency to monthly
                except:
                    df_ = df_.resample('MS').ffill() 

                    
            elif freq=="weekly":
                try:
                    df_.index.freq = 'W-MON'  # Manually setting frequency to monthly
                except:
                    df_ = df_.resample('W-MON').ffill() 
                
            else:
                try:
                    df_.index.freq = 'D'  # Manually setting frequency to monthly
                except:
                    df_ = df_.resample('D').ffill() 

            decomposition = seasonal_decompose(df_[var_to_plot], model='additive')
            fig = decomposition.plot()
            for ax in fig.axes:
                plt.sca(ax)  # set current axis
                plt.xticks(rotation=45)  # rotate ticks
            
            plt.tight_layout()  # adjust subplots to fit into figure area.
            plt.show()
        except Exception:
            print("ALERT: no trend decomposition due to small sample")
            time.sleep(5)
            pass

        
    
    @staticmethod
    def plot_MA(df,var_to_plot):
        rolmean = df[var_to_plot].rolling(6).mean()
        rolstd =df[var_to_plot].rolling(6).std()
        plt.figure(figsize=(22,10))   
        orig = plt.plot(df["date"],df[var_to_plot], color='red',label=f'{var_to_plot}')
        mean = plt.plot(df["date"],rolmean, color='black', label=f'Rolling Mean of {var_to_plot}')
        std = plt.plot(df["date"],rolstd, color='green', label = f'Rolling Std of {var_to_plot}')
        plt.xlabel("Date")
        plt.ylabel(var_to_plot)
        plt.title('Evolution, Mean & Standard Deviation Plot')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_autocorr(df,var):
        df_=df[var]
        plot_acf(df_.dropna())
        plt.show()

    @staticmethod
    def plot_partial_autocorr(df,var):
        df_=df[var]
        plot_pacf(df_.dropna())
        plt.show()


    @staticmethod
    def plot_statistics_by_numeric(df,var_use,vars_to_plot_against): 
        if df.shape[0] > 100:
            plt.hexbin(df[var_use], df[vars_to_plot_against], gridsize=20, cmap='viridis')
            plt.colorbar(label='count in bin')
            plt.xlabel(f'Variable {var_use}')
            plt.ylabel(f'Variable {vars_to_plot_against}')
            plt.show()
        else:
                
            #Scatter_plots
            plt.scatter(df[var_use], df[vars_to_plot_against])
            plt.xlabel(f'Variable {var_use}')
            plt.ylabel(f'Variable {vars_to_plot_against}')
            plt.show()




    
