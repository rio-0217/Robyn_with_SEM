# install.packages("remotes") # Install remotes first if you haven't already
# remotes::install_github("facebookexperimental/Robyn/R")
library(Robyn)
# # install.packages("dplyr")
# library(dplyr)
# # install.packages("tidyverse")
# library(tidyverse)
# install.packages("glmnet")
# library(regsem)

# Please, check if you have installed the latest version before running this demo. Update if not
# https://github.com/facebookexperimental/Robyn/blob/main/R/DESCRIPTION#L4
packageVersion("Robyn")

# ## Force multicore when using RStudio
Sys.setenv(R_FUTURE_FORK_ENABLE = "true")
options(future.fork.enable = TRUE)

## Must install the python library Nevergrad once

# install.packages("reticulate") # Install reticulate first if you haven't already
library("reticulate") # Load the library

# ## Option 1: nevergrad installation via PIP (no additional installs)
virtualenv_create("r-reticulate")
use_virtualenv("r-reticulate", required = TRUE)
py_install("nevergrad", pip = TRUE)
py_install("semopy", pip = TRUE)
py_install("pandas", pip = TRUE)
py_install("numpy", pip = TRUE)
py_install("graphviz", pip = TRUE)
py_install("pydot", pip = TRUE)
# py_config() # Check your python version and configurations
# In case nevergrad still can't be installed,
Sys.setenv(RETICULATE_PYTHON = "~/.virtualenvs/r-reticulate/bin/python")
# Reset your R session and re-install Nevergrad with option 1

# install.packages("dplyr")
library(dplyr)
# install.packages("tidyverse")
library(tidyverse)
## Option 2: nevergrad installation via conda (must have conda installed)
# conda_create("r-reticulate", "Python 3.9") # Only works with <= Python 3.9 sofar
# use_condaenv("r-reticulate")
# conda_install("r-reticulate", "nevergrad", pip=TRUE)
# py_config() # Check your python version and configurations
## In case nevergrad still can't be installed,
## please locate your python file and run this line with your path:
# use_python("~/Library/r-miniconda/envs/r-reticulate/bin/python3.9")
# Alternatively, force Python path for reticulate with this:
# Sys.setenv(RETICULATE_PYTHON = "~/Library/r-miniconda/envs/r-reticulate/bin/python3.9")
# Finally, reset your R session and re-install Nevergrad with option 2

################################################################
#### Step 1: Load data

## Check simulated dataset or load your own dataset
data("dt_simulated_weekly")
head(dt_simulated_weekly)

## Check holidays from Prophet
data("dt_prophet_holidays")
head(dt_prophet_holidays)

# Directory where you want to export results to (will create new folders)
robyn_object <- "~/Desktop/robyn_poc"

InputCollect <- robyn_inputs(
  dt_input = dt_simulated_weekly,
  dt_holidays = dt_prophet_holidays,
  date_var = "DATE", # date format must be "2020-01-01"
  dep_var = "revenue", # there should be only one dependent variable
  dep_var_type = "revenue", # "revenue" (ROI) or "conversion" (CPA)
  prophet_vars = c("trend", "season", "holiday"), # "trend","season", "weekday" & "holiday"
  prophet_country = "DE", # input one country. dt_prophet_holidays includes 59 countries by default
  context_vars = c("competitor_sales_B"), # e.g. competitors, discount, unemployment etc
  paid_media_spends = c("tv_S", "ooh_S", "print_S", "facebook_S", "search_S"), # mandatory input
  paid_media_vars = c("tv_S", "ooh_S", "print_S", "facebook_I", "search_clicks_P"), # mandatory.
  # paid_media_vars must have same order as paid_media_spends. Use media exposure metrics like
  # impressions, GRP etc. If not applicable, use spend instead.
  organic_vars = "newsletter", # marketing activity without media spend
  # factor_vars = c("events"), # force variables in context_vars or organic_vars to be categorical
  window_start = "2016-11-21",
  window_end = "2018-08-20",
  adstock = "geometric" # geometric, weibull_cdf or weibull_pdf.
)

hyperparameters <- list(
  facebook_S_alphas = c(0.5, 3),
  facebook_S_gammas = c(0.3, 1),
  facebook_S_thetas = c(0, 0.3),
  print_S_alphas = c(0.5, 3),
  print_S_gammas = c(0.3, 1),
  print_S_thetas = c(0.1, 0.4),
  tv_S_alphas = c(0.5, 3),
  tv_S_gammas = c(0.3, 1),
  tv_S_thetas = c(0.3, 0.8),
  search_S_alphas = c(0.5, 3),
  search_S_gammas = c(0.3, 1),
  search_S_thetas = c(0, 0.3),
  ooh_S_alphas = c(0.5, 3),
  ooh_S_gammas = c(0.3, 1),
  ooh_S_thetas = c(0.1, 0.4),
  newsletter_alphas = c(0.5, 3),
  newsletter_gammas = c(0.3, 1),
  newsletter_thetas = c(0.1, 0.4)
)

InputCollect <- robyn_inputs(InputCollect = InputCollect, hyperparameters = hyperparameters)

source("./Robyn_with_SEM/SEMRobyn/R/R/auxiliary.R")
source("./Robyn_with_SEM/SEMRobyn/R/R/checks.R")
source("./Robyn_with_SEM/SEMRobyn/R/R/convergence.R")
source("./Robyn_with_SEM/SEMRobyn/R/R/imports.R")
source("./Robyn_with_SEM/SEMRobyn/R/R/exports.R")
source("./Robyn_with_SEM/SEMRobyn/R/R/json.R")
source("./Robyn_with_SEM/SEMRobyn/R/R/refresh.R")
source("./Robyn_with_SEM/SEMRobyn/R/R/response.R")
source("./Robyn_with_SEM/SEMRobyn/R/R/transformation.R")
source("./Robyn_with_SEM/SEMRobyn/R/R/zzz.R")
library(dplyr)
library(doParallel)
library(doRNG)
library(tidyverse)
library(lares)
library(rPref)
library(utils)
library(parallel)
library(patchwork)
library(jsonlite)
library(foreach)
library(minpack.lm)
library(glmnet)
library(prophet)
library(stats)
library(stringr)
library(tidyr)
library(nloptr)
library(lubridate)
library(ggridges)


model <- '
  # measurement model
    interest =~ dep_var
    awareness =~ dep_var
    
  # regressions
    interest ~ facebook_S + search_S
    awareness ~ tv_S + ooh_S + print_S + newsletter
    dep_var ~ competitor_sales_B
    dep_var ~ holiday
    dep_var ~ season
    dep_var ~ trend
    
  # residual correlations
    search_S ~~ facebook_S
    tv_S ~~ ooh_S
    tv_S ~~ print_S
    tv_S ~~ newsletter
    ooh_S ~~ print_S
    ooh_S ~~ newsletter
    print_S ~~ newsletter
'

source("./Robyn_with_SEM/SEMRobyn/R/R/model_v2.R")
source_python("./Robyn_with_SEM/SEMRobyn/R/R/sem_model.py")
OutputModels <- robyn_run(
  InputCollect = InputCollect, # feed in all model specification
  # cores = 4, # default to max available
  # add_penalty_factor = FALSE, # Untested feature. Use with caution.
  iterations = 2000,# recommended for the dummy dataset with no calibration
  trials = 5, # 5 recommended for the dummy dataset
  outputs = FALSE, # outputs = FALSE disables direct model output - robyn_outputs()
  use_SEM = TRUE,#TRUE,
  SEM_mod = model,
  robyn_object = "./semplot"
)

## Check MOO (multi-objective optimization) convergence plots
OutputModels$convergence$moo_distrb_plot
OutputModels$convergence$moo_cloud_plot

## Calculate Pareto optimality, cluster and export results and plots. See ?robyn_outputs
OutputCollect <- robyn_outputs(
  InputCollect, OutputModels,
  # pareto_fronts = "auto",
  # calibration_constraint = 0.1, # range c(0.01, 0.1) & default at 0.1
  csv_out = "pareto", # "pareto", "all", or NULL (for none)
  clusters = TRUE, # Set to TRUE to cluster similar models by ROAS. See ?robyn_clusters
  plot_pareto = TRUE, # Set to FALSE to deactivate plotting and saving model one-pagers
  plot_folder = robyn_object, # path for plots export
  export = TRUE # this will create files locally
)

select_model <- "1_170_2"
# Run ?robyn_allocator to check parameter definition
# Run the "max_historical_response" scenario: "What's the revenue lift potential with the
# same historical spend level and what is the spend mix?"
AllocatorCollect1 <- robyn_allocator(
  InputCollect = InputCollect,
  OutputCollect = OutputCollect,
  select_model = select_model,
  scenario = "max_historical_response",
  export = TRUE,
  date_min = "2016-11-21",
  date_max = "2018-08-20"
)
