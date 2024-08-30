from dataclasses import dataclass
import pandas as pd
import yaml
import time

from .extraction import ExtractData
from. preprocessing import Preprocessing
from .EDA import EDA
from .feature_engineering import TsIntoTabular
from .feature_engineering import feature_engineering
from .predictions import SetFunctions
from .model_selector import ModelSelector
from .result_analysis import ResultAnalysis


@dataclass
class Pipeline:
    config_path:str
    ruta_padre:str
    def __post_init__(self):
        
        
        with open(fr"{self.ruta_padre}/assets/configs/{self.config_path}") as yam:
            self.config=yaml.load(yam,Loader=yaml.FullLoader)
        self.config["ruta_padre"]=self.ruta_padre
        self.config["differenciate"]=True
        

    def extract_df_by_type(self):
        if self.config["data_extraction"]["type_extraction"]=="local_csv":

            self.df_original=ExtractData.extract_df_csv(self.config)

        elif self.config["data_extraction"]["type_extraction"]=="local_parquet":
            self.df_original=ExtractData.extract_df_parquet(self.config)

        elif self.config["data_extraction"]["type_extraction"]=="local_excel":
            self.df_original=ExtractData.extract_df_excel(self.config)

        elif self.config["data_extraction"]["type_extraction"]=="s3_csv":
            self.df_original=ExtractData.extract_df_s3_csv(self.config)

        elif self.config["data_extraction"]["type_extraction"]=="s3_parquet":
            self.df_original=ExtractData.extract_df_s3_parquet(self.config)
        elif self.config["data_extraction"]["type_extraction"]=="s3_excel":
            self.df_original=ExtractData.extract_df_s3_parquet(self.config)



        self.df_original=ExtractData.clean_duplicated_columns_rows(self.df_original,self.config)

        self.df_original=ExtractData.clean_specific_vars_writing_issues(self.df_original,self.config)

        self.df_original=ExtractData.get_needed_info_by_type(self.df_original,self.config)
        
        

    def preprocessing_specific_df(self):

        df_date_correct=Preprocessing.transform_date(self.df_original,self.config)

        df_correct_time=Preprocessing.get_time_groups(df_date_correct,self.config)

        df_grouped=Preprocessing.group_by_time(df_correct_time,self.config)

        df_grouped=Preprocessing.fill_empty_time(df_grouped,self.config)

        df_grouped=Preprocessing.take_out_na(df_grouped,self.config)

        df_grouped=Preprocessing.divide_specific_columns_by_time(df_grouped,self.config)        
        
        self.df_ts=Preprocessing.move_one_back(df_grouped,self.config)
        

        



    def get_diffs(self):#TODO: quitar ifs, los dejamos por si no quieres hacer differencias
        
        if (self.config["preprocessing"]["difference"]=="one") | (self.config["preprocessing"]["difference"]=="second"):
            df_diff=TsIntoTabular.get_difference(self.df_ts,self.config,self.config["data_extraction"]["prediction_column"],"difference")
            if self.config["preprocessing"]["difference"]=="second":
                df_diff=TsIntoTabular.get_difference(df_diff,self.config,"difference","secdifference")
        else:
            df_diff=self.df_ts
            
        self.df_diff=df_diff

        
    def get_best_diff(self):

        result_var=EDA.check_stat_adfuller(self.df_diff,self.config["data_extraction"]["prediction_column"])

        result_diff=1
        result_secdiff=1
        if (self.config["preprocessing"]["difference"]=="one") | (self.config["preprocessing"]["difference"]=="second"):

            result_diff=EDA.check_stat_adfuller(self.df_diff,"difference")
            if self.config["preprocessing"]["difference"]=="second":

                result_secdiff=EDA.check_stat_adfuller(self.df_diff,"secdifference")
        
        if result_var<0.05:
            self.config["var_in_use"]=self.config["data_extraction"]["prediction_column"]
            
            if len(self.config["preprocessing"]["columns_to_shift"])>=1:
                self.df_diff=Preprocessing.drop_first_row_by_grp(self.df_diff,self.config)
                
        elif result_diff<0.05:
            self.config["var_in_use"]="difference"
            self.df_diff=Preprocessing.drop_first_row_by_grp(self.df_diff,self.config)

        elif result_secdiff<0.05:
            self.config["var_in_use"]="secdifference"
            
            self.df_diff=Preprocessing.drop_first_row_by_grp(Preprocessing.drop_first_row_by_grp(self.df_diff,self.config),self.config)

        else:
            self.config["var_in_use"]=self.config["data_extraction"]["prediction_column"]
            if len(self.config["preprocessing"]["columns_to_shift"])>=1:
                self.df_diff=Preprocessing.drop_first_row_by_grp(self.df_diff,self.config)
            
            

        
        
        self.df_save=self.df_diff.copy()

        
        
        if self.config["preprocessing"]["normalize_if_not_diff"]:
            if self.config["var_in_use"]==self.config["data_extraction"]["prediction_column"]:
                    if self.config["preprocessing"]["type_normalization"]=="z_score":
                        result=Preprocessing.standardize(self.df_diff[self.config["var_in_use"]])
                        
                    elif self.config["preprocessing"]["type_normalization"]=="robust_scaler":
                        result=Preprocessing.robust_scale(self.df_diff[self.config["var_in_use"]])
                    else:
                        result=Preprocessing.min_max_scaler(self.df_diff[self.config["var_in_use"]])
                    
                    self.config["1st_parameter_norm"]={}
                    self.config["2nd_parameter_norm"]={}
                    
                    self.df_diff[self.config["var_in_use"]],self.config["1st_parameter_norm"][self.config["var_in_use"]],self.config["2nd_parameter_norm"][self.config["var_in_use"]]=result


        else:
            if self.config["preprocessing"]["normalize"]:
                

                    if self.config["preprocessing"]["type_normalization"]=="z_score":
                        result=Preprocessing.standardize(self.df_diff[self.config["var_in_use"]])
                        
                    elif self.config["preprocessing"]["type_normalization"]=="robust_scaler":
                        result=Preprocessing.robust_scale(self.df_diff[self.config["var_in_use"]])
                    else:
                        result=Preprocessing.min_max_scaler(self.df_diff[self.config["var_in_use"]])
                    
                    self.config["1st_parameter_norm"]={}
                    self.config["2nd_parameter_norm"]={}
                    
                    self.df_diff[self.config["var_in_use"]],self.config["1st_parameter_norm"][self.config["var_in_use"]],self.config["2nd_parameter_norm"][self.config["var_in_use"]]=result
        
                    
        
        
        
        if self.config["preprocessing"]["normalize_independent_var"]:
                df_static=self.df_diff.copy()
                df_static=df_static.drop(columns=[self.config["var_in_use"],"difference","secdifference"],errors="ignore")
                
                self.config["1st_parameter_norm_exo"]={}
                self.config["2nd_parameter_norm_exo"]={}
                
                for var in (list(df_static.select_dtypes(include=[int, float]).columns)):
                        if self.config["preprocessing"]["type_normalization"]=="z_score":
                            
                            self.df_diff[var],self.config["1st_parameter_norm"][var],self.config["2nd_parameter_norm_exo"][var]=Preprocessing.standardize(self.df_diff[var])
                                
                        elif self.config["preprocessing"]["type_normalization"]=="robust_scaler":
                            self.df_diff[var],self.config["1st_parameter_norm_exo"][var],self.config["2nd_parameter_norm_exo"][var]=Preprocessing.robust_scale(self.df_diff[var])
                        else:
                            self.df_diff[var],self.config["1st_parameter_norm_exo"][var],self.config["2nd_parameter_norm_exo"][var]=Preprocessing.min_max_scaler(self.df_diff[var])


    def get_lags(self):
        
        self.config["ts_into_tab"]["steps"]=TsIntoTabular.compute_optimal_lags(self.df_diff,self.config,self.config["var_in_use"])

        self.df_lag=TsIntoTabular.get_lag(self.df_diff,self.config,self.config["var_in_use"])

        if self.config["type_analysis"]!="ts_only":
            for variable in self.config["ts_into_tab"]["vars_to_lag"]:
                self.df_lag=TsIntoTabular.get_lag(self.df_lag,self.config,variable)



    def plot_everything_eda(self):

        EDA.variable_heatmap(self.df_eng_save,self.config,False)

        for var in [self.config["data_extraction"]["prediction_column"],"difference","secdifference"]:
            if var in self.df_lag.columns:
                EDA.decompose_ts(self.df_lag,var,self.config["preprocessing"]["aggr_time_level"])


        EDA.plot_autocorr(self.df_lag,self.config["var_in_use"]) 
        EDA.plot_partial_autocorr(self.df_lag,self.config["var_in_use"])
        EDA.variable_heatmap(self.df_eng_save,self.config,True)

        for var in self.config["EDA"]["vars_to_plot_scatter"]:
            EDA.plot_statistics_by_numeric(self.df_lag,var[0],var[1])

        
    def add_engineer_variables(self):
        
        
        
        if self.config["feature_eng"]["add_holidays"]:
            self.df_ts_model=feature_engineering.add_holidays_feature(self.df_lag,self.config["feature_eng"]["country"],self.config["feature_eng"]["years_list"],self.config["preprocessing"]["aggr_time_level"])
        else:
            self.df_ts_model=self.df_lag
            
        if self.config["feature_eng"]["add_date"]: 
            self.df_eng=feature_engineering.add_time_variable(self.df_ts_model,self.config)
        
        else:
            self.df_eng=self.df_ts_model.copy()

        for var_nin_nfin in self.config["feature_eng"]["mean_vars"]: 
            self.df_eng=feature_engineering.get_mean_variable(self.df_eng,var_nin_nfin[0],var_nin_nfin[1],var_nin_nfin[2])
        for var_nin_nfin in self.config["feature_eng"]["mean_weighted_vars"]: 
            self.df_eng=feature_engineering.get_mean_weighthed_variable(self.df_eng,var_nin_nfin[0],var_nin_nfin[1],var_nin_nfin[2])

        for var_nin_nfin in self.config["feature_eng"]["median_vars"]:
            self.df_eng=feature_engineering.get_median_variable(self.df_eng,var_nin_nfin[0],var_nin_nfin[1],var_nin_nfin[2])
        
        self.df_eng_save=self.df_eng.copy()
        if self.config["var_in_use"]=="difference":

            self.df_eng=self.df_eng.drop([self.config["data_extraction"]["prediction_column"],"secdifference"],errors="ignore")
        elif self.config["var_in_use"]=="secdifference":

            self.df_eng=self.df_eng.drop([self.config["data_extraction"]["prediction_column"],"difference"],errors="ignore")
        elif self.config["var_in_use"]==self.config["data_extraction"]["prediction_column"]:

            self.df_eng=self.df_eng.drop(["difference","secdifference"],errors="ignore")

        

    
        
    def get_models(self):


        self.baselines_models=SetFunctions.get_all_preds_baselines(self.df_save,self.config)

        self.tab_models=SetFunctions.get_all_preds_tab(self.df_eng,self.config)

        
        if self.config["model"]["use_feature_eng_on_ts_models"]:
            if self.config["model"]["use_lags_on_ts_models"]:
                self.ts_models=SetFunctions.get_all_preds_ts(self.df_eng,self.config)
            else:
                columns_to_use=[col for col in self.df_eng.columns if "lag" not in col]
                self.ts_models=SetFunctions.get_all_preds_ts(self.df_eng[columns_to_use],self.config)
        else:
            if self.config["model"]["use_lags_on_ts_models"]:
                self.ts_models=SetFunctions.get_all_preds_ts(self.df_ts_model,self.config)
            else:
                columns_to_use=[col for col in self.df_ts_model.columns if "lag" not in col]
                self.ts_models=SetFunctions.get_all_preds_ts(self.df_ts_model[columns_to_use],self.config)
        


        
        if self.config["model"]["use_original_for_baseline"]:
            self.df_pred=pd.concat([self.ts_models,self.tab_models],axis=1)

        else:
            self.df_pred=pd.concat([self.ts_models,self.tab_models,self.baselines_models],axis=1)
        

        #We only have to normalize the results, we dont care about independent variables as no annalysis is required
        if self.config["preprocessing"]["normalize_if_not_diff"]:
            if self.config["var_in_use"]==self.config["data_extraction"]["prediction_column"]:
                for model in list(self.df_pred.columns):
                    if self.config["preprocessing"]["type_normalization"]=="z_score":
                        self.df_pred[model]=Preprocessing.destandardize(self.df_pred[model],self.config["1st_parameter_norm"][self.config["var_in_use"]],self.config["2nd_parameter_norm"][self.config["var_in_use"]])
                    elif self.config["preprocessing"]["type_normalization"]=="robust_scaler":
                        self.df_pred[model]=Preprocessing.robust_denormalize(self.df_pred[model],self.config["1st_parameter_norm"][self.config["var_in_use"]],self.config["2nd_parameter_norm"][self.config["var_in_use"]])
                    else:
                        self.df_pred[model]=Preprocessing.de_min_max_scaler(self.df_pred[model],self.config["1st_parameter_norm"][self.config["var_in_use"]],self.config["2nd_parameter_norm"][self.config["var_in_use"]])

                
        else:
            if self.config["preprocessing"]["normalize"]:
                for model in list(self.df_pred.columns):
                    if self.config["preprocessing"]["type_normalization"]=="z_score":
                        self.df_pred[model]=Preprocessing.destandardize(self.df_pred[model],self.config["1st_parameter_norm"][self.config["var_in_use"]],self.config["2nd_parameter_norm"][self.config["var_in_use"]])
                    elif self.config["preprocessing"]["type_normalization"]=="robust_scaler":
                        self.df_pred[model]=Preprocessing.robust_denormalize(self.df_pred[model],self.config["1st_parameter_norm"][self.config["var_in_use"]],self.config["2nd_parameter_norm"][self.config["var_in_use"]])
                    else:
                        self.df_pred[model]=Preprocessing.de_min_max_scaler(self.df_pred[model],self.config["1st_parameter_norm"][self.config["var_in_use"]],self.config["2nd_parameter_norm"][self.config["var_in_use"]])

                

    def get_preds_back_diff(self):
        rows_to_pred=int(self.config["model"]["percentage_predicted"])
        models_pred=list(self.df_pred.columns)
        if self.config["var_in_use"]=="difference":
            
            
            
            var_initial=self.df_save[self.config["data_extraction"]["prediction_column"]].shift(1)
            self.df_pred["var_comp"]=var_initial.tail(rows_to_pred).reset_index(drop=True)
            for model in models_pred:
                self.df_pred[model]=self.df_pred["var_comp"]+self.df_pred[model]

            self.df_pred=self.df_pred.drop(columns=["var_comp"])
            

        elif self.config["var_in_use"]=="secdifference":
            diff_var=self.df_save["difference"].shift(1)

            self.df_pred["diff_comp"]=diff_var.tail(rows_to_pred).reset_index(drop=True)

            for model in models_pred:
                self.df_pred[model]=self.df_pred["diff_comp"]+self.df_pred[model]            
            
            var_initial=self.df_save[self.config["data_extraction"]["prediction_column"]].shift(1)

            self.df_pred["var_comp"]=var_initial.tail(rows_to_pred).reset_index(drop=True)

            for model in models_pred:
                self.df_pred[model]=self.df_pred["var_comp"]+self.df_pred[model]


            self.df_pred=self.df_pred.drop(columns=["var_comp","diff_comp"])
        
        if self.config["model"]["use_original_for_baseline"]:
            self.df_pred=pd.concat([self.df_pred,self.baselines_models],axis=1)
            
        df_bagg=SetFunctions.get_bagging_models(self.df_pred,list(self.ts_models.columns),list(self.tab_models.columns),list(self.baselines_models.columns))
        df_weight_bagg=SetFunctions.get_weighted_bagging_models(self.df_pred,list(self.ts_models.columns),list(self.tab_models.columns),list(self.baselines_models.columns),self.df_save,self.config)
        
        self.df_pred=pd.concat([self.df_pred,df_bagg,df_weight_bagg],axis=1)

        
    def get_best_model(self): 
        
        self.df_all_info=pd.concat([self.df_pred,self.df_save[self.config["data_extraction"]["prediction_column"]].tail(len(self.df_pred)).reset_index(drop=True)],axis=1)
        
        df_to_analyse=self.df_all_info.iloc[(self.config["model_selector"]["min_steps"]):].reset_index(drop=True).copy()
        df_to_analyse["mdl_slct_pred"]=0.0
        df_to_analyse["slct_name"]=None
        

        for step in range(len(df_to_analyse)):
            df_to_use=self.df_all_info.iloc[step:(self.config["model_selector"]["min_steps"]+step)]

            if self.config["model_selector"]["exclude_bagging_models"]:
                not_bagg_columns= [col for col in df_to_use.columns if 'bagg' not in col]
                df_to_use=df_to_use[not_bagg_columns]
            mdl_slct=ModelSelector(df_to_use,self.config)

            error_table=mdl_slct.get_error_table()
                
            best_error,best_model=mdl_slct.get_best(error_table)

            df_to_analyse.loc[step,"mdl_slct_pred"]=float(self.df_pred.iloc[self.config["model_selector"]["min_steps"]+step][best_model])
            df_to_analyse.loc[step,"model_selector_name"]=best_model
        
        
        try:
            print()        
            print()        
            print("Model amount in model selector: ")        
            print(df_to_analyse["model_selector_name"].value_counts())
        except Exception:
            print("No model selector prediction")
        time.sleep(5)
        
        self.df_to_analyse=df_to_analyse
        

    def get_metrics(self):
        analyser=ResultAnalysis(self.df_to_analyse,self.config)
        
        metrics_df=analyser.get_metrics()

        ResultAnalysis.show_df(metrics_df)
        
        
        summary_df=ResultAnalysis.get_conclusions(self.df_to_analyse,metrics_df,list(self.baselines_models.columns),list(self.tab_models.columns),list(self.ts_models.columns))
        
        ResultAnalysis.save_df(metrics_df,summary_df,self.df_all_info,self.config)




