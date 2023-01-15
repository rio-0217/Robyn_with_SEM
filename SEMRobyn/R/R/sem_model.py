import numpy as np
import pandas as pd
# import sklearn
import semopy
import os
import datetime as dt
# import warnings
# warnings.simplefilter('ignore')

def regsem_model(
  path, 
  data, 
  lambda_scaled,
  out_dir,
  out_label):
  
  dt_now=dt.datetime.now()
  date=dt_now.strftime("%Y%m%d")
  model = semopy.Model(path)
  model.fit(
    data,
    regularization = semopy.create_regularization(
        model = model,
        regularization ='l2-naive',
        c = lambda_scaled))
  # os.makedirs(f"{out_dir}/semplot_{date}",exist_ok=True)
  # semopy.semplot(model, f"{out_dir}/semplot_{date}/structure{out_label}.png")

  return model

def regsem_model_refit(
  model,
  path,
  x_train, y_train, x_val, y_val, x_test, y_test
):
  df_int = 1

  y_trainPred = model.predict(x_train)
  rsq_train = get_rsq_py(y_Pred=y_trainPred, y=y_train, p=x_train.shape[0], df_int=df_int, n_train=len(y_train))
  # y_pred = rsq_train
  nrmse_train = np.sqrt(np.mean((y_train - y_trainPred["dep_var"])**2)) / (np.max(y_train) - np.min(y_train))
  
  try:
    y_valPred = model.predict(x_val)
    rsq_val = get_rsq_py(y_Pred=y_valPred, y=y_val, p=x_val.shape[0], df_int=df_int, n_train=len(y_val))
    nrmse_val = np.sqrt(np.mean((y_val - y_valPred["dep_var"])**2)) / (np.max(y_val) - np.min(y_val))
  except:
    y_valPred = 0
    rsq_val = 0
    nrmse_val = 0
  # y_pred = rsq_train+rsq_val
    
  try:
    y_testPred = model.predict(x_test)
    rsq_test = get_rsq_py(y_Pred=y_testPred, y=y_test, p=x_test.shape[0], df_int=df_int, n_train=len(y_test))
    nrmse_test = np.sqrt(np.mean((y_test - y_testPred["dep_var"])**2)) / (np.max(y_test) - np.min(y_test))
  except:
    y_testPred = 0
    rsq_test = 0
    nrmse_test = 0
    
  y_pred = rsq_train+rsq_val+rsq_test
    
  # nrmse_train = np.sqrt(np.mean((y_train - y_trainPred["dep_var"])**2)) / (np.max(y_train) - np.min(y_train))
  coef_dt = model.inspect(std_est=True)

  mod_out = {
    "rsq_train":rsq_train, 
    "rsq_val":rsq_val,
    "rsq_test":rsq_test,
    "nrmse_train":nrmse_train, 
    "nrmse_val":nrmse_val,
    "nrmse_test":nrmse_test,
    "mod":model,
    "df_int":df_int,
    "coef_dt":coef_dt,
    "y_train_pred":y_trainPred,
    "y_val_pred":y_valPred,
    "y_test_pred":y_testPred,
    "y_pred":y_pred
    }
  return mod_out
  
def get_rsq_py(y_Pred, y, p, df_int, n_train):
  try:
    sse = np.sum((y_Pred["dep_var"] - y)**2)
  
    sst = np.sum((y - np.mean(y))**2)
    
    rsq = 1 - sse / sst 
    rsq_out = rsq
  
    # if (p is not None & df_int is not None):
    if n_train >0:
      n = n_train # for oos dataset, use n from train set for adj. rsq
    else:
      n = len(y)
      
    rdf = n - p - 1
    rsq_adj = 1 - (1 - rsq) * ((n - df_int) / rdf)
    rsq_out = rsq_adj
  except:
    rsq_out = 0
   
  return rsq_out

# def semplot(mod: Model, filename: str, inspection=None, plot_covs=False,
#             plot_exos=True, images=None, engine='dot', latshape='circle',
#             plot_ests=True, std_ests=False, show=False):
#     """
#     Draw a SEM diagram.
# 
#     Parameters
#     ----------
#     mod : Model
#         Model instance.
#     filename : str
#         Name of file where to plot is saved.
#     inspection : pd.DataFrame, optional
#         Parameter estimates as returned by Model.inspect(). The default is
#         None.
#     plot_covs : bool, optional
#         If True, covariances are also drawn. The default is False.
#     plot_exos: bool, optional
#         If False, exogenous variables are not plotted. It might be useful,
#         for example, in GWAS setting, where a number of exogenous variables,
#         i.e. genetic markers, is oblivious. Has effect only with ModelMeans or
#         ModelEffects. The default is True.
#     images : dict, optional
#         Node labels can be replaced with images. It will be the case if a map
#         variable_name->path_to_image is provided. The default is None.
#     engine : str, optional
#         Graphviz engine name to use. The default is 'dot'.
#     latshape : str, optional
#         Graphviz-compaitable shape for latent variables. The default is
#         'circle'.
#     plot_ests : bool, optional
#         If True, then estimates are also plotted on the graph. The default is
#         True.
#     std_ests : bool, optional
#         If True and plot_ests is True, then standardized values are plotted
#         instead. The default is False.
#     show : bool, optional
#         If True, the 
# 
#     Returns
#     -------
#     Graphviz graph.
# 
#     """
#     if not __GRAPHVIZ:
#         raise ModuleNotFoundError("No graphviz module is installed.")
#     if type(mod) is str:
#         mod = Model(mod)
#     if not hasattr(mod, 'last_result'):
#         plot_ests = False
#     if inspection is None:
#         inspection = mod.inspect(std_est=std_ests)
#     if images is None:
#         images = dict()
#     if std_ests:
#         inspection['Estimate'] = inspection['Est. Std']
#     t = filename.split('.')
#     filename, ext = '.'.join(t[:-1]), t[-1]
#     g = graphviz.Digraph('G', format=ext, engine=engine)
#     
#     g.attr(overlap='scale', splines='true')
#     g.attr('edge', fontsize='12')
#     g.attr('node', shape=latshape, fillcolor='#cae6df', style='filled')
#     for lat in mod.vars['latent']:
#         if lat in images:
#             g.node(lat, label='', image=images[lat])
#         else:
#             g.node(lat, label=lat)
#     
#     g.attr('node', shape='box', style='')
#     for obs in mod.vars['observed']:
#         if obs in images:
#             g.node(obs, label='', image=images[obs])
#         else:
#             g.node(obs, label=obs)
# 
#     regr = inspection[inspection['op'] == '~']
#     all_vars = mod.vars['all']
#     try:
#         exo_vars = mod.vars['observed_exogenous']
#     except KeyError:
#         exo_vars = set()
#     for _, row in regr.iterrows():
#         lval, rval, est = row['lval'], row['rval'], row['Estimate']
#         if (rval not in all_vars) or (~plot_exos and rval in exo_vars) or\
#             (rval == '1'):
#             continue
#         if plot_ests:
#             pval = row['p-value']
#             label = '{:.3f}'.format(float(est))
#             if pval !='-':
#                 label += r'\np-val: {:.2f}'.format(float(pval))
#         else:
#             label = str()
#         g.edge(rval, lval, label=label)
#     if plot_covs:
#         covs = inspection[inspection['op'] == '~~']
#         for _, row in covs.iterrows():
#             lval, rval, est = row['lval'], row['rval'], row['Estimate']
#             if lval == rval:
#                 continue
#             if plot_ests:
#                 pval = row['p-value']
#                 label = '{:.3f}'.format(float(est))
#                 if pval !='-':
#                     label += r'\np-val: {:.2f}'.format(float(pval))
#             else:
#                 label = str()
#             g.edge(rval, lval, label=label, dir='both', style='dashed')
#     g.render(filename, view=show)
#     return g

