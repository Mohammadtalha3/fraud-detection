from fastapi import FastAPI
from pydantic import BaseModel

import uvicorn
import pandas as pd
import util
import data_pipeline as data_pipeline
import preprocessing as preprocessing
import modelling as modelling

config_data = util.load_config()

ohe_data = util.pickle_load(config_data["ohe_path"])
le_data = util.pickle_load(config_data["le_path"])
le_label = util.pickle_load(config_data["le_label_path"])
model_data = util.pickle_load(config_data["production_model_path"])

class api_data(BaseModel):
    #policy_bind_date : str
    #incident_date : str
    months_as_customer : int
    age : int
    policy_number : int
    policy_annual_premium : int
    insured_zip : int
    capital_gains : int
    capital_loss : int
    incident_hour_of_the_day : int
    total_claim_amount : int
    injury_claim : int
    property_claim : int
    vehicle_claim : int
    policy_deductable : str
    umbrella_limit : str
    number_of_vehicles_involved : str
    bodily_injuries : str
    witnesses : str
    auto_year : str
    policy_state : str
    policy_csl : str
    insured_sex : str
    insured_hobbies : str
    incident_type : str  
    collision_type : str
    incident_severity : str
    authorities_contacted : str
    incident_state : str
    incident_city : str
    property_damage : str
    police_report_available : str
    auto_make : str
    auto_model : str

app = FastAPI()

@app.get("/")
def home():
    return "Hello, FastAPI up!"


def transform_frontend_data(frontend_data, backend_columns):
    # Create a DataFrame with all backend columns, initialized to 0
    transformed_data = pd.DataFrame(0, index=[0], columns=backend_columns)
    
    # Copy over the values from frontend_data
    for col in frontend_data:
        if col in backend_columns:
            transformed_data[col] = frontend_data[col]
    
    return transformed_data

@app.post("/predict/")
def predict(data: api_data):
    # 0. Load config
    config_data = util.load_config()

    #print('this is api data', data)
    
    # 1. Convert data api to dataframe
    data = pd.DataFrame(data).set_index(0).T.reset_index(drop=True)

    #data1= data[config_data['datetime_columns']].copy()
    #print('this is datatime columns', data1)

    print('this is api data', data.values)
    print('this is api data', data.columns.tolist())

    # 2. Convert Dtype
    data = data_pipeline.type_data(data)
    data.columns = config_data['api_predictor']

    print('after selecting predictor', data.values)
    print('after selecting predictor', data.columns.tolist())


    # 3. Check range data
    try:
        data_pipeline.check_data(data, config_data)
    except AssertionError as ae:
        return {"res": [], "error_msg": str(ae)}

    # 4. Split data into predictor and label
    data = data[config_data["predictor"]].copy()

    print('after selecting predictor after split', data.values)
    print('after selecting predictor after split', data.columns.tolist())
    
    

    # 5. Split data into numerical and categorical for handling each type of data
    data_num, data_cat = preprocessing.splitNumCat(data)

    print('after selecting predictor after split numcat', data_num.values)
    print('after selecting predictor after split numcat', data_num.columns.tolist())

    print('after selecting predictor after split numcat', data_cat.values)
    print('after selecting predictor after split numcat', data_cat.columns.tolist())

    print('this is data numerical we are passing to num', data_num.columns.tolist())

    # 6. Imputed numerical data for any missing value
    data_num_imputed, imputer_num = preprocessing.imputerNum(data = data_num)

    # 7. Imputed Categorical data for any missing value
    print('this is data num after  imp', data_num_imputed.columns.tolist())
    print('this is the num after imp', data_num_imputed.values)
    data_cat_imputed, imputer_cat = preprocessing.imputerCat(data = data_cat)

    print('this is data cat after  imp', data_cat_imputed.columns.tolist())
    print('this is the cat after imp', data_cat_imputed.values)

    # 8. Encoding data categorical using OHE for nominal data and LE for ordinal data
    data_cat_ohe, encoder_ohe_col, encoder_ohe = preprocessing.OHEcat(data = data_cat_imputed)
    data_cat_le, encoder_le = preprocessing.LEcat(data = data_cat_imputed)

    # 9. Concatenate ohe and le encoded data
    data_cat_concat = pd.concat([data_cat_ohe,data_cat_le], axis = 1)

    print('this is data cat after cnct', data_cat_concat.columns.tolist())
    print('this is the cat after cnct', data_cat_concat.values)
    print('this is the cat after cnct', len(data_cat_concat.columns))



    #print('this is contatced daat cat', data_cat_concat.columns.tolist())

    # 10. Concatenate numerical data and categorical data
    data_concat = pd.concat([data_num_imputed, data_cat_concat], axis=1)

    print('this is the data contacts', data_concat.values)
    print('this is the data name contacts', data_concat.columns.tolist())
    print('len of the data passing to std', len(data_concat.columns))

    backend_columns = config_data['backend_columns']

    data_transformed = transform_frontend_data(data_concat, backend_columns)

    # 11. Standardize value of train data
    data_clean, scaler = preprocessing.standardizeData(data = data_transformed)

    print('this is the data aftr std', data_clean.values)
    print('this is the data aftr std', data_clean.columns.tolist())
    
    # 12. Load x_train variable to equalize new data len columns with model fit len columns
    x_train, y_train = modelling.load_train_clean(config_data)

    print('this is train data in api', x_train)
    
    # 13. Equalize the columns since OHE create 131 columns, with non existing value must have value = 0

    print('this is data_clean before', data_clean.values)
    print('this is data_clean before', data_clean.columns.tolist())
    
    if len(data_clean.columns) != 131:
        print('this is len of cols',len(data_clean.columns))
        d_col = set(data_clean.columns).symmetric_difference(set(x_train['nonbalance'].columns))
        
        for col in d_col:
            data_clean[col] = 0
    
    #print('this is after len', len(data_clean.columns))
    
    print('this is data_clean after', data_clean)

    # 13. Predict the data
    y_pred = model_data["model_data"]["model_object"].predict(data_clean)

    print('this is the y_pred val', y_pred)

    if y_pred[0] == 0:
        y_pred = "TIDAK FRAUD."
    else:
        y_pred = "FRAUD."

    return {"res" : y_pred, "error_msg": ""}

if __name__ == "__main__":
    uvicorn.run("api:app", host = "127.0.0.1", port = 8080)