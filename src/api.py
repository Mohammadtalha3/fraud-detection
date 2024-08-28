from fastapi import FastAPI
from pydantic import BaseModel
import random

import uvicorn
import pandas as pd
import util
import data_pipeline as data_pipeline
import preprocessing as preprocessing
import modelling as modelling
import numpy as np

config_data = util.load_config()

ohe_data = util.pickle_load(config_data["ohe_path"])
le_data = util.pickle_load(config_data["le_path"])
le_label = util.pickle_load(config_data["le_label_path"])
model_data = util.pickle_load(config_data["production_model_path"])
cat_imputer= util.pickle_load(config_data['imputer_cat'])
num_imputer= util.pickle_load(config_data['imputer_num'])

print('this is model objedt', model_data['model_data']['model_object'])
#print('this is model objedt', model_data['model_object'])

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
    data_num_imputed, imputer_num = preprocessing.imputerNum(data = data_num, imputer=num_imputer)

    # 7. Imputed Categorical data for any missing value
    print('this is data num after  imp', data_num_imputed.columns.tolist())
    print('this is the num after imp', data_num_imputed.values)
    print('making sure the lenght of the data', len(data_num_imputed.values))
    data_cat_imputed, imputer_cat = preprocessing.imputerCat(data = data_cat, imputer=cat_imputer)

    print('this is data cat after  imp', data_cat_imputed.columns.tolist())
    print('this is the cat after imp', data_cat_imputed.values)

    encoder_columns=['policy_state', 'policy_csl', 'policy_deductable', 'insured_sex', 'insured_hobbies',
                      'collision_type', 'authorities_contacted', 'incident_state', 'incident_city',
      'property_damage', 'police_report_available', 'auto_make', 'auto_model']
    enc_col=data_cat_imputed[encoder_columns]

    encoder_columns= ohe_data.get_feature_names_out(enc_col.columns)

    ordinal = ['incident_type','witnesses','incident_severity','auto_year','umbrella_limit','bodily_injuries',
            'number_of_vehicles_involved']
    data_le=data_cat_imputed[ordinal]

    print('this is data lee', data_le.columns.tolist())

    #print('These are  the columns for encoder', encoder_columns.columns.tolist())

    # 8. Encoding data categorical using OHE for nominal data and LE for ordinal data
    data_cat_ohe, encoder_ohe_col, encoder_ohe = preprocessing.OHEcat(data = data_cat_imputed,encoder_col=encoder_columns, encoder=ohe_data)
    data_cat_le, encoder_le = preprocessing.LEcat(data = data_le, encoder=le_data)

    print('this is Categorical_ohe', data_cat_ohe.columns.tolist())
    print('this is Categorical_ohe', data_cat_ohe.values)
    
    print('this is Categorical_le', data_cat_le.columns.tolist())
    print('this is Categorical_le', data_cat_le.values)

   


    scaler=util.pickle_load(config_data['standardizer_file'][0])
    if scaler:
         print('yes file is loaded')

    print('this is scaller data loading ', scaler)
     
    



    # 9. Concatenate ohe and le encoded data
    data_cat_concat = pd.concat([data_cat_ohe,data_cat_le], axis = 1)

    print('this is data cat after cnct', data_cat_concat.columns.tolist())
    print('this is the cat after cnct', data_cat_concat.values)
    print('this is the cat after cnct', len(data_cat_concat.columns))



    #print('this is contatced daat cat', data_cat_concat.columns.tolist())

    # 10. Concatenate numerical data and categorical data
    

    # print('this is the data contacts', data_concat.values)
    # print('this is the data name contacts', data_concat.columns.tolist())
    # print('len of the data passing to std', len(data_concat.columns))

    #backend_columns = config_data['backend_columns']

    #data_transformed = transform_frontend_data(data_clean, backend_columns)
    

    # 11. Standardize value of train data
    data_clean, _ = preprocessing.standardizeData(data = data_num_imputed, scaler=scaler)

    print('this is the data aftr std', data_clean.values)
    print('this is the data aftr std', data_clean.columns.tolist())
    
    data_clean = pd.concat([data_clean,data_cat_concat], axis=1)

    print('this is api data clean', data_clean.columns.tolist())
    print('this is the data column after we have ', data_clean.values)
    print('this is the data columns len', len(data_clean.columns))
    print('this is complete data ', data_clean)


    

    #data_clean= pd.concat(data_clean,data_concat)
    
    # 12. Load x_train variable to equalize new data len columns with model fit len columns
    #x_train, y_train = modelling.load_valid_clean(config_data)

   
    #x_train, y_train = modelling.load_valid_clean(config_data)

    
    #testing the data train load/// already tested x_valid

    # x_train, y_train = modelling.load_train_clean(config_data)
    # x_train = {key: [value] if not isinstance(value, (list, np.ndarray, pd.Series)) else value 
    #            for key, value in x_train.items()}
    # x_train = pd.DataFrame(x_train)
    # train_columns = x_train.columns.tolist()

    # # Reorder columns to match the training data
    # data_clean = data_clean.reindex(columns=train_columns, fill_value=0)
    

    # 13. Equalize the columns since OHE create 131 columns, with non existing value must have value = 0

    print('this is data_clean before', data_clean.values)
    print('this is data_clean before', data_clean.columns.tolist())
    print('this is data_clean before', len(data_clean.columns))



    x_train, y_train = modelling.load_train_clean(config_data)


    

    # print('this is x_train data', x_train['nonbalance'].columns.tolist())
    # print('this is x_train data', set(x_train['nonbalance'].columns.tolist()))
    # print('this is x_train dataclean', set(data_clean.columns.tolist()))
    # print('this is data_clean lenght ', data_clean['number_of_vehicles_involved'].values)

    
    if len(data_clean.columns) != 131:
        print('this is len of cols',len(data_clean.columns))
         #d_col = set(data_clean.columns).symmetric_difference(set(x_train['nonbalance'].columns))
        d_col = set(x_train['nonbalance'].columns) - set(data_clean.columns)
        print('this is d_Col after adjustment', d_col)

        d_col_list = list(d_col)

        missing_cols_df = pd.DataFrame(0, index=data_clean.index, columns=d_col_list)
        
        # # Concatenate the missing columns DataFrame with the original DataFrame
        data_clean = pd.concat([data_clean, missing_cols_df], axis=1)

        print('this is clean data before setting to 0', data_clean.columns.tolist())
        print('this is clean data before setting to 0', data_clean.values)

        
        
      
        for col in d_col:
           
            print('this is cols to be zerod', col)

            print(data_clean[col])

            data_clean[col] = 0
        print('this is in the data clean work', data_clean.columns.tolist())
        print('these are values we set to 0 ', data_clean['number_of_vehicles_involved'].values)
             

    
    #columns_to_use= x_train['nonbalance'].columns.tolist()

    #data_clean = data_clean.reindex(columns=columns_to_use, fill_value=0)

    #print('Here is the data',data_clean)
        
    
    print('this is the data we are getting aftter setting zer0', data_clean.columns.tolist())
    print('this is the data we are getting aftter setting zer0', data_clean.values)


    #train_columns = x_train.columns.tolist()
    
    #print('this is after len', len(data_clean.columns))
    #print('this is the x_train nonbal', x_train['nonbalance'].columns)
    #print('this is the x_train nonbal', x_train['oversampling'].columns)
    #print('this is the x_train nonbal', x_train['smote'].columns)
    
    #print('this is data_clean after', data_clean.values)
    #print('this is data_clean after', len(data_clean.columns))
    #print('this is data_clean after', data_clean.columns.tolist())

    

    #print('this id reindexed data', data_clean.columns.tolist())
    
    
  

    #model_data = util.pickle_load(config_data["production_model_path"])
    # 13. Predict the data
    print('This is the model_data in the api.py',model_data['model_data'])

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

# Print the DataFrame
    print(data_clean)


    y_pred = model_data["model_data"]["model_object"].predict(data_clean)

    print('this is the y_pred val', y_pred)

    if y_pred[0] :
        y_pred = "NOTFRAUD."
        
    else:

        #y_pred= random.choice(['FRAUD', 'NOT FRAUD'])        
        y_pred = "FRAUD."  



    return {"res" : y_pred, "error_msg": ""}

if __name__ == "__main__":
    uvicorn.run("api:app", host = "127.0.0.1", port = 8080)