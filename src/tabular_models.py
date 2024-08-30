from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR 
import xgboost as xgb
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import  GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
import warnings
class TbModels:
    @staticmethod
    def xgboost(df,df_future,config):



        if config["model"]["xgb"]["optimize_params"]:
            model=TbModels.optimize_xgboost(df,config)
            

        else:
            parametros=config["model"]["xgb"]["xgb_params"]

            xgb_model = xgb.XGBRegressor(**parametros)


            model =xgb_model.fit( df[config["tab_exo_columns"]], df[config["var_in_use"]])

        forecast = model.predict(df_future[config["tab_exo_columns"]])

        return [forecast[0],"xgb"]
    
    @staticmethod
    def optimize_xgboost(df,config):
        xg_reg = xgb.XGBRegressor(objective='reg:squarederror', seed=42)
        X=df[config["tab_exo_columns"]]
        y=df[config["var_in_use"]]
        # Define the parameter grid
        param_grid = config["model"]["xgb"]["xgb_param_grid"]

        # Create the GridSearchCV object
        grid_search = GridSearchCV(estimator=xg_reg, param_grid=param_grid, 
                                scoring='neg_mean_squared_error', 
                                cv=3, verbose=0, n_jobs=-1)

        # Fit the grid search to the data
        grid_search.fit(X, y)

        if config["model"]["xgb"]["optimize_only_once"]:
                config["model"]["xgb"]["optimize_params"]=False
                config["model"]["xgb"]["xgb_params"]=grid_search.best_params_

        return  grid_search.best_estimator_

    @staticmethod
    def linear_regress(df,df_future,config):

        regr = LinearRegression()
        

        regr.fit(df[config["tab_exo_columns"]], df[config["var_in_use"]])

        forecast = regr.predict(df_future[config["tab_exo_columns"]])

        return [forecast[0],"linear"]


    @staticmethod
    def random_forest(df,df_future,config):

        if config["model"]["rf"]["optimize_params"]:
                        
            param_grid= config["model"]["rf"]["rf_param_grid"]

            X=df[config["tab_exo_columns"]]
            y=df[config["var_in_use"]]

            grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=0 , n_jobs=-1)

            # Fit GridSearchCV to the data
            grid_search.fit(X, y)

            # Get the best parameters and best estimator
            if config["model"]["rf"]["optimize_only_once"]:
                config["model"]["rf"]["optimize_params"]=False
                config["model"]["rf"]["rf_params"]=grid_search.best_params_                 


            forecast =  grid_search.best_estimator_.predict(df_future[config["tab_exo_columns"]])

            return [forecast[0],f"rf"]
        else:

            rf_model = RandomForestRegressor(**config["model"]["rf"]["rf_params"])
            

            rf_model.fit(df[config["tab_exo_columns"]], df[config["var_in_use"]])

            forecast = rf_model.predict(df_future[config["tab_exo_columns"]])

            return [forecast[0],"rf"]
    
    @staticmethod
    def svm(df,df_future,config): #We use couse of radial kernel, linear and polynomial already applied
        if config["model"]["svr"]["optimize_params"]:
                        
            param_grid= config["model"]["svr"]["svr_param_grid"]

            X=df[config["tab_exo_columns"]]
            y=df[config["var_in_use"]]

            grid_search = GridSearchCV(estimator=SVR(), param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=0, n_jobs=-1)

            # Fit GridSearchCV to the data
            grid_search.fit(X, y)

            # Get the best parameters and best estimator
            if config["model"]["svr"]["optimize_only_once"]:
                config["model"]["svr"]["optimize_params"]=False
                config["model"]["svr"]["svr_params"]=grid_search.best_params_                      


            forecast =  grid_search.best_estimator_.predict(df_future[config["tab_exo_columns"]])

            return [forecast[0],f"svr"]
        else:

            regr = SVR(**config["model"]["svr"]["svr_params"])
            

            regr.fit(df[config["tab_exo_columns"]], df[config["var_in_use"]])

            forecast = regr.predict(df_future[config["tab_exo_columns"]])

            return [forecast[0],"svr"]

    @staticmethod
    def lasso_regress(df,df_future,config):

        regr = Lasso(alpha=1,max_iter=100000,tol=0.001)
        

        regr.fit(df[config["tab_exo_columns"]], df[config["var_in_use"]])

        forecast = regr.predict(df_future[config["tab_exo_columns"]])

        return [forecast[0],"lasso"]

    @staticmethod
    def ridge_regress(df,df_future,config):

        regr = Ridge(alpha=1,max_iter=100000,tol=0.001)
        

        regr.fit(df[config["tab_exo_columns"]], df[config["var_in_use"]])

        forecast = regr.predict(df_future[config["tab_exo_columns"]])

        return [forecast[0],"ridge"]
    
    @staticmethod
    def elastic_regress(df,df_future,config):

        if config["model"]["elastic"]["optimize_params"]:
                        
            param_grid= config["model"]["elastic"]["elastic_param_grid"]

            X=df[config["tab_exo_columns"]]
            y=df[config["var_in_use"]]
            warnings.filterwarnings("ignore", category=ConvergenceWarning)

            grid_search = GridSearchCV(estimator=ElasticNet(), param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=0, n_jobs=-1)

            # Fit GridSearchCV to the data
            grid_search.fit(X, y)

            # Get the best parameters and best estimator
            if config["model"]["elastic"]["optimize_only_once"]:
                config["model"]["elastic"]["optimize_params"]=False
                config["model"]["elastic"]["elastic_params"]=grid_search.best_params_                      


            forecast =  grid_search.best_estimator_.predict(df_future[config["tab_exo_columns"]])

            return [forecast[0],f"elastic"]
        else:

            regr = ElasticNet(**config["model"]["elastic"]["elastic_params"])
            

            regr.fit(df[config["tab_exo_columns"]], df[config["var_in_use"]])

            forecast = regr.predict(df_future[config["tab_exo_columns"]])

            return [forecast[0],"elastic"]



    @staticmethod
    def knn(df,df_future,config):
        if config["model"]["knn"]["optimize_params"]:
            model=TbModels.optimize_knn(df,config)

        else:
            parametros=config["model"]["knn"]["knn_params"]

            knn = KNeighborsRegressor(**parametros)

            model = knn.fit(df[config["tab_exo_columns"]],  df[config["var_in_use"]])

        forecast=model.predict(df_future[config["tab_exo_columns"]])

        return [forecast[0],"knn"]
    @staticmethod
    def optimize_knn(df,config):
        param_grid = config["model"]["knn"]["knn_param_grid"]  # Example: Testing k from 1 to 30
        knn = KNeighborsRegressor()
        X=df[config["tab_exo_columns"]]
        y=df[config["var_in_use"]]

        # Set up GridSearchCV
        grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)

        # Fit the model
        grid_search.fit(X, y)
        if config["model"]["knn"]["optimize_only_once"]:
            config["model"]["knn"]["optimize_params"]=False
            config["model"]["knn"]["knn_params"]=grid_search.best_params_       
        
        return  grid_search.best_estimator_
    @staticmethod
    def bagging_tab():
        return