
import sys
import os
import pandas as pd
import logging
import warnings
import time

sys.path.append(os.getcwd())
sys.path.append("../../")
sys.path.append("../")
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.model_pipeline import Pipeline 


warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="prophet")
warnings.filterwarnings("ignore", category=FutureWarning, message="'M' is deprecated")
warnings.filterwarnings("ignore", message="Optimization failed to converge")
warnings.filterwarnings("ignore", message="Objective did not converge")
warnings.filterwarnings("ignore", message="No supported index is available")
warnings.filterwarnings("ignore", message="An unsupported index")
warnings.filterwarnings('ignore', message='DataFrame is highly fragmented')

def main(config_path,ruta_padre):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None) 
    logging.getLogger('prophet').setLevel(logging.ERROR)
    logging.getLogger('concurrent.futures').setLevel(logging.CRITICAL)

    process=Pipeline(config_path=config_path,ruta_padre=ruta_padre)

    process.extract_df_by_type()

    process.preprocessing_specific_df()

    process.get_diffs()  

    process.get_best_diff()  

    process.get_lags() 

    process.add_engineer_variables()

    process.plot_everything_eda()
    
    start_time = time.time()
 
    process.get_models()  
    
    end_time = time.time()
      
    process.get_preds_back_diff()    

    process.get_best_model()

    process.get_metrics()


    elapsed_time = end_time - start_time
    
    print(f"Models Execution time: {elapsed_time}")

if __name__ =="__main__":
    config_path="sp500.yaml"
    ruta_padre="C:/Users/Alex Mayo/Documents/GitHub/TFG"
    main(config_path,ruta_padre)