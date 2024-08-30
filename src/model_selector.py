class ModelSelector:
    def __init__(self,table,config) -> None:
        self.table=table
        self.config=config

    def get_error_table(self):
        numeric_columns=list(self.table.select_dtypes(include=["int","float"]).columns)

        error_table=self.table[numeric_columns]

        for preds in list(numeric_columns):
            if preds!=self.config["data_extraction"]["prediction_column"]:
                error_table[f"{preds}"]=error_table[f"{preds}"]-error_table[self.config["data_extraction"]["prediction_column"]]

        self.error_table=error_table

        return self.error_table

    def get_best(self,error_table):
        model_columns=list(error_table.columns)
        try:
            model_columns.remove(self.config["data_extraction"]["prediction_column"])
        except Exception:  
            pass
        model_columns
        if self.config["type_analysis"]=="ts_categorical":
            grp_error=error_table.groupby(self.config["preprocessing"]["aggr_lvl"])[model_columns].mean().reset_index(drop=True)

            grp_error_lower_error=grp_error[model_columns].min(axis=1)
            grp_error_best_model=grp_error[model_columns].idxmin(axis=1)

            return grp_error_lower_error , grp_error_best_model
        else:
            grp_error=self.error_table[model_columns].abs().mean()

            grp_error_min=grp_error[model_columns].min()
            grp_error_name=grp_error[model_columns].idxmin()

            return float(grp_error_min),grp_error_name

    @staticmethod
    def closest_to_zero(row):
        return min(row,key=abs)


