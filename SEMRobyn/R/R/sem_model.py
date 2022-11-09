import numpy as np
import pandas as pd
# import sklearn
import semopy

def regsem_model(
  path, 
  data, 
  lambda_scaled):
  model = semopy.Model(path)
  model.fit(
    data,
    regularization = semopy.create_regularization(
        model = model,
        regularization ='l2-naive',
        c = lambda_scaled))
  return model

def regsem_model_refit(
  model,
  path,
  x_train, 
  y_train, 
  lambda_scaled
):
  y_trainPred = model.predict(x_train)
  sse = np.sum((y_trainPred["dep_var"] - y_train)**2)

  sst = np.sum((y_train - np.mean(y_train))**2)
  rsq_train = 1 - sse / sst
  df_int = 1
  p = x_train.shape[0]
  if p>0:
    n = len(y_train)
    rdf = n - p - 1
    rsq_train = 1 - (1 - rsq_train) * ((n - df_int) / rdf)
    
  nrmse_train = np.sqrt(np.mean((y_train - y_trainPred["dep_var"])**2)) / (np.max(y_train) - np.min(y_train))
  coef_dt = model.inspect(std_est=True)

  mod_out = {
    "rsq_train":rsq_train, 
    "nrmse_train":nrmse_train, 
    "y_pred":y_trainPred["dep_var"],
    "mod":model,
    "df_int":df_int,
    "coef_dt":coef_dt}
  return mod_out
  
  
