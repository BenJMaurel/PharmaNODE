# Title: Tacrolimus Pharmacokinetic Analysis from Command Line
# Author: jbw (converted for command-line execution)
# Date: 2025-09-09

# -----------------------------------------------------------------------------
# 1. SETUP: Load Libraries & Parse Command-Line Arguments
# -----------------------------------------------------------------------------

# --- Load Libraries ---
# Ensure all required packages are installed first by running in your R console:
# install.packages(c("argparse", "tidyverse", "mrgsolve", "mapbayr", "MESS"))
suppressPackageStartupMessages(library(argparse))
suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(mrgsolve))
suppressPackageStartupMessages(library(mapbayr))
suppressPackageStartupMessages(library(MESS))
suppressPackageStartupMessages(library(lixoftConnectors))
library(glue)
suppressPackageStartupMessages(library(furrr)) # For parallel processing
# --- Define and Parse Command-Line Arguments ---
parser <- ArgumentParser(description = "Run mapbayr analysis for Tacrolimus PK data.")
parser$add_argument("--virtual_cohort", type = "character", required = TRUE,
                    help = "Path to the virtual cohort CSV file (e.g., virtual_cohort.csv)")
parser$add_argument("--output_dir", type = "character", default = ".",
                    help = "Directory to save the output PDF and CSV files [default: current directory]")
parser$add_argument("--cores", type = "integer", default = 1,
                    help = "Number of CPU cores to use for parallel processing [default: 1]")
parser$add_argument("--monolix_path", type = "character", required = TRUE,
                    help = "Path to the Monolix installation directory")
parser$add_argument("--model_file", type = "character", required = TRUE,
                    help = "Path to the Monolix model file (e.g., model.txt)")
args <- parser$parse_args()

# --- Initialize Lixoft Connectors ---
message("Initializing Lixoft Connectors...")
initializeLixoftConnectors(software = 'monolix', path = args$monolix_path)

# --- Construct Paths ---
data_file_path    <- file.path(args$virtual_cohort)
project_save_path <- file.path(args$output_dir, "tacro_monolix_project.mlxtran")
model_file_path   <- args$model_file

message(paste("Data file:", data_file_path))
message(paste("Model file:", model_file_path))
message(paste("Project will be saved to:", project_save_path))
column_mapping <- c(
  ID          = "id",         # Subject Identifier
  TIME        = "time",       # Time of measurement or dose
  DV = "observation",         # Dependent Variable (the measurement)
  AMT      = "amount",        # Dose Amount
  CYP   = "catcov",        # Covariate for Genotype 
  # ss          = "ss",         # Steady-State flag
  # II          = "ii",         # Inter-dose Interval
  ST   = "catcov"         # Covariate for Status
  # HT   = "contcov"          # Covariate for Hematocrit
)

newProject(
  modelFile = model_file_path,
  data = list(
    dataFile = data_file_path,
    headerTypes = column_mapping )
)
setIndividualParameterModel(list( correlationBlocks = list(id = list()), 
                                  covariateModel = list(CL = c(CYP = TRUE))) )

setIndividualParameterModel(list( correlationBlocks = list(id = list()), 
                                  covariateModel = list(Vc = c(ST = TRUE))) )

setIndividualParameterModel(list( correlationBlocks = list(id = list()), 
                                  covariateModel = list(KTR = c(ST = TRUE))) )
                                  
# Define how the columns in your data file map to Monolix data types.
# You MUST adjust the names on the left ('ID', 'time', 'y') to match the
# actual column headers in your data.csv file.

cat("-> Project created and data loaded successfully.\n")

# setPopulationParameterInformation(CL_pop = list(initialValue = 21.2), 
#                                   Vc_pop  = list(initialValue = 486),
#                                   Q_pop = list(initialValue = 79),
#                                   Vp_pop = list(initialValue = 271.0),
#                                   KTR_pop = list(initialValue = 3.34),
#                                   beta_CL_CYP_1 = list(initialValue = log(2.0)),
#                                   beta_Vc_ST_1 = list(initialValue = log(0.29)),
#                                   beta_KTR_ST_1 = list(initialValue = log(1.53)))

cat("-> Initial parameter values have been set.\n")

saveProject(projectFile = project_save_path)
cat(paste("-> Project configured and saved to:", project_save_path, "\n"))

runPopulationParameterEstimation()

cat("-> SAEM estimation has been started!\n")
cat("You can monitor its progress in the Monolix GUI or check the results folder that was created next to your project file.\n")

# 5. Extract the results!
# Get population parameters (the final estimates for TVCL, omega_CL, etc.)
monolix_results <- getEstimatedPopulationParameters()
results_df <- data.frame(
  Parameter = names(monolix_results),
  Value     = unname(monolix_results),
  row.names = NULL # Clean up row names
)

# Just print the data frame to the console
print(results_df)
# -----------------------------------------------------------------------------
# 2. MODEL DEFINITION
# -----------------------------------------------------------------------------
message("Defining the tacrolimus pharmacokinetic model...")

# code_tac <- "
# [PROB]
# # Population pharmacokinetics of tacrolimus
# # Woillard de Winter BJCP 2011 

# [PARAM] @annotated
# TVCL : 21.2 : Typical value of clearance (L/h)
# TVV1 : 486 : Typical apparent central volume of distribution (L)
# TVQ : 79 : Typical intercomp clearance 1 (L/h)
# TVV2 : 271 : Typical peripheral volume of distribution (L)
# TVKTR : 3.34 : Typical transfer rate constant (1/h)
# HTCL : -1.14 : Effect of hematocrit on clearance
# CYPCL : 2.00 : Effect of CYP on clearance
# STKTR : 1.53 : Effect of study on KTR
# STV1 : 0.29 : Effect of study on V1


# ETA1 : 0 : ETA on clearance
# ETA2 : 0 : ETA on V1
# ETA3 : 0 : ETA on Q
# ETA4 : 0 : ETA on V2
# ETA5 : 0 : ETA on KTR

# $PARAM @annotated @covariates
# HT : 35 : Hematocrit (percentage)
# ST : 1 : Prograf (1) adv (0)
# CYP : 0 : Expressor (1) non-expressor (0)


# [CMT] @annotated
# DEPOT : Dosing compartment (mg) [ADM]
# TRANS1 : Transit compartment 1 (mg)
# TRANS2 : Transit compartment 2 (mg)
# TRANS3 : Transit compartment 3 (mg)
# CENT : Central compartment (mg) [OBS]
# PERI : Peripheral compartment (mg)

# [OMEGA]
# 0.08
# 0.10
# 0.29
# 0.36
# 0.06

# [MAIN]


# // Apparent clearance (CL_app) and other PK parameters
# double CL_app = TVCL * pow(HT / 35, HTCL) * pow(CYPCL, CYP) * exp(ETA1 + ETA(1));
# double V1_app = TVV1  * pow(STV1, ST) * exp(ETA2 + ETA(2));

# // Other parameters
# double Q = TVQ * exp(ETA3 + ETA(3));
# double V2 = TVV2 * exp(ETA4 + ETA(4));
# double KTR = TVKTR * pow(STKTR, ST) * exp(ETA5 + ETA(5));

# [SIGMA] @annotated
# PROP : 0.012 : Proportional residual unexplained variability
# ADD : 0.5 : Additive residual unexplained variability

# [ODE]
# dxdt_DEPOT = -KTR * DEPOT;
# dxdt_TRANS1 = KTR * DEPOT - KTR * TRANS1;
# dxdt_TRANS2 = KTR * TRANS1 - KTR * TRANS2;
# dxdt_TRANS3 = KTR * TRANS2 - KTR * TRANS3;
# dxdt_CENT = KTR * TRANS3 - (CL_app + Q) * CENT / V1_app + Q * PERI / V2;
# dxdt_PERI = Q * CENT / V1_app - Q * PERI / V2;

# [TABLE]
# double CONC = CENT / (V1_app / 1000);
# capture DV = CONC * (1 + PROP) + ADD;
# $CAPTURE DV CL_app 
# "
# mod_tac <- mcode("tac_model", code_tac)

omega_Cl <- monolix_results[["omega_CL"]]^2
omega_Vc <- monolix_results[["omega_Vc"]]^2
omega_Q <- monolix_results[["omega_Q"]]^2
omega_Vp <- monolix_results[["omega_Vp"]]^2
omega_KTR <- monolix_results[["omega_KTR"]]^2
prop <- monolix_results[["b"]]
add <- monolix_results[['a']]
# omega_Cl <- 0.3
# omega_Vc <- 0.2
# omega_Q <- 0.3
# omega_Vp <- 0.2
# omega_KTR <- 0.3

code_tac <- glue(
"
[PROB]
# Population pharmacokinetics of tacrolimus
# Woillard de Winter BJCP 2011

[PARAM] @annotated
TVCL : 21.2 : Typical value of clearance (L/h)
TVV1 : 486 : Typical apparent central volume of distribution (L)
TVQ : 79 : Typical intercomp clearance 1 (L/h)
TVV2 : 271 : Typical peripheral volume of distribution (L)
TVKTR : 3.34 : Typical transfer rate constant (1/h)
HTCL : -1.14 : Effect of hematocrit on clearance
CYPCL : 2.00 : Effect of CYP on clearance
STKTR : 1.53 : Effect of study on KTR
STV1 : 0.29 : Effect of study on V1

ETA1 : 0 : ETA on clearance
ETA2 : 0 : ETA on V1
ETA3 : 0 : ETA on Q
ETA4 : 0 : ETA on V2
ETA5 : 0 : ETA on KTR

$PARAM @annotated @covariates
HT : 35 : Hematocrit (percentage)
ST : 1 : Prograf (1) adv (0)
CYP : 0 : Expressor (1) non-expressor (0)

[CMT] @annotated
DEPOT : Dosing compartment (mg) [ADM]
TRANS1 : Transit compartment 1 (mg)
TRANS2 : Transit compartment 2 (mg)
TRANS3 : Transit compartment 3 (mg)
CENT : Central compartment (mg) [OBS]
PERI : Peripheral compartment (mg)

[OMEGA]
{omega_Cl}
{omega_Vc}
{omega_Q}
{omega_Vp}
{omega_KTR}

[MAIN]
// Apparent clearance (CL_app) and other PK parameters
double CL_app = TVCL * pow(CYPCL, CYP) * exp(ETA1 + ETA(1));
double V1_app = TVV1 * pow(STV1, ST) * exp(ETA2 + ETA(2));

// Other parameters
double Q = TVQ * exp(ETA3 + ETA(3));
double V2 = TVV2 * exp(ETA4 + ETA(4));
double KTR = TVKTR * pow(STKTR, ST) * exp(ETA5 + ETA(5));

[SIGMA] @annotated
PROP : {prop} : Proportional residual unexplained variability
ADD : {add} : Additive residual unexplained variability

[ODE]
dxdt_DEPOT = -KTR * DEPOT;
dxdt_TRANS1 = KTR * DEPOT - KTR * TRANS1;
dxdt_TRANS2 = KTR * TRANS1 - KTR * TRANS2;
dxdt_TRANS3 = KTR * TRANS2 - KTR * TRANS3;
dxdt_CENT = KTR * TRANS3 - (CL_app + Q) * CENT / V1_app + Q * PERI / V2;
dxdt_PERI = Q * CENT / V1_app - Q * PERI / V2;

[TABLE]
double CONC = CENT / (V1_app)*1000;
capture DV = CONC * (1 + PROP) + ADD;
$CAPTURE DV CL_app V1_app Q V2 KTR
"
)
mod_tac <- mcode("tac_model", code_tac)

mod_tac_updated <- param(
  mod_tac,
  # Fixed Effects
  TVCL  = monolix_results[["CL_pop"]],
  TVV1  = monolix_results[["Vc_pop"]],
  TVQ   = monolix_results[["Q_pop"]],
  TVV2  = monolix_results[["Vp_pop"]],
  TVKTR = monolix_results[["KTR_pop"]],
  # Covariate Effects
  # HTCL  = monolix_results[["beta_CL_HT"]],
  CYPCL = exp(monolix_results[["beta_CL_CYP_1"]]),
  STKTR = exp(monolix_results[["beta_KTR_ST_1"]]),
  STV1  = exp(monolix_results[["beta_Vc_ST_1"]])
)

# prop_variance <- monolix_results[["prop_err_pop"]]^2
# add_variance  <- monolix_results[["add_err_pop"]]^2

# # Create a diagonal SIGMA matrix with the VARIANCES
# sigma_matrix <- diag(c(prop_variance, add_variance))

# # Use smat() to explicitly update the SIGMA block
# mod_tac_updated <- smat(mod_tac_updated, sigma_matrix)

# Update the OMEGA block (no changes here)
# omega_variances <- c(
#   monolix_results[["omega_T1_CL"]]^2,
#   monolix_results[["omega_T1_Vc"]]^2,
#   monolix_results[["omega_Q"]]^2,
#   monolix_results[["omega_Vp"]]^2,
#   monolix_results[["omega_T1_KTR"]]^2
# )

# cat("\n--- The final omega_variances vector ---\n")
# omega_matrix <- diag(omega_variances)
# mod_tac_updated <- omat(mod_tac_updated, mat = omega_matrix)

# ---------------------------------------------------------------------------
# ## STEP 4: VERIFY AND USE THE UPDATED MODEL
# ---------------------------------------------------------------------------
cat("\n--- Verifying Updated mrgsolve Model ---\n")
print(param(mod_tac_updated))

# -----------------------------------------------------------------------------
# 3. DATA PREPARATION
# -----------------------------------------------------------------------------
message(paste("Loading data from:", args$virtual_cohort))
raw_data <- read_csv(args$virtual_cohort, na = c("null", ".", "NA", ""), trim_ws = TRUE, show_col_types = FALSE)
# --- Create the 3-point dataset for estimation ---
# This dataset will contain the 3 observations for every patient
obs_data_to_process <- raw_data %>%
  # Keep only observation rows for this step
  filter(is.na(AMT)) %>%
  # Use a more robust filter for the time points
  filter(
    near(TIME, 0) |
    between(TIME, 0.8, 1.2) |
    between(TIME, 2.4, 3.6)
  )

# -----------------------------------------------------------------------------
# 4. ANALYSIS: MAPBAYESIAN ESTIMATION AND AUC CALCULATION
# -----------------------------------------------------------------------------
message("Starting analysis for each patient...")
if(args$cores > 1) {
  message(paste("Using", args$cores, "cores for parallel processing."))
  plan(multisession, workers = args$cores)
} else {
  plan(sequential) # Use this for no parallel processing
}
# --- Define function to run analysis for one patient ID ---
run_one_id <- function(patient_id, all_raw_data, obs_data) {
  # Get the 3 observation rows for this specific patient
  df_obs <- obs_data %>% filter(ID == patient_id)
  if(nrow(df_obs) != 3) {
    warning(paste("ID", patient_id, "did not have 3 valid observations. Skipping."))
    return(NULL)
  }
  
  # Get the single dosing row for this patient to extract covariates
  df_dose <- all_raw_data %>% filter(ID == patient_id, !is.na(AMT), TIME == 0)
  if(nrow(df_dose) != 1) {
    warning(paste("ID", patient_id, "did not have one valid dosing row. Skipping."))
    return(NULL)
  }
  
  # Extract values
  amt_val <- df_dose$AMT
  drug_val <- df_dose$DRUG
  cyp_val <- df_dose$CYP
  auc_obs <- df_dose$AUC
  st_val  <- if_else(drug_val == "Advagraf", 0, 1)
  ii_val  <- if_else(drug_val == "Advagraf", 24, 12)
  
  # Build the estimation dataset using the verified pipeable workflow
  est_obj <- mod_tac_updated %>%
    adm_rows(time = 0, amt = amt_val, ss = 1, ii = ii_val, addl = 4) %>%
    add_covariates(CYP = cyp_val, ST = st_val) %>%
    obs_rows(time = df_obs$TIME[1], DV = df_obs$DV[1]) %>%
    obs_rows(time = df_obs$TIME[2], DV = df_obs$DV[2]) %>%
    obs_rows(time = df_obs$TIME[3], DV = df_obs$DV[3]) %>%
    mapbayest(verbose = FALSE)

  # Augment and calculate AUC
  auc_start <- 0
  auc_end <- if (st_val == 1) 12 else 24
  aug <- mapbayr::augment(est_obj, start = auc_start, end = auc_end, delta = 0.1)
  
  ipred_win <- aug$aug_tab %>%
    filter(type == "IPRED", dplyr::between(time, 0, auc_end)) %>%
    transmute(time, DV = value)

  ipred_win_unique <- ipred_win %>%
  distinct(time, .keep_all = TRUE)
  
  auc_ipred <- auc(ipred_win_unique$time, ipred_win_unique$DV)
  # Return a list containing both the plot object and the results tibble
  ind_params <- get_param(est_obj, .name = c("CL_app", "V1_app", "Q", "V2", "KTR"))
  # --- MODIFICATION END ---
  cat(ind_params['V'])
  # Return a list containing both the plot object and the results tibble
  list(
    # MODIFICATION: Added xlim = c(0, 24) to set the plot's time range
    plot = plot(aug, main = sprintf("ID %s — amt=%g mg — ii=%gh — ST=%s",
                                      patient_id, amt_val, ii_val, as.character(st_val)),
                xlim = c(0, 24)), 
    # --- MODIFICATION START ---
    # Add the extracted parameters to the results tibble.
    results = tibble(
      ID = patient_id, ST = st_val, amt = amt_val, ii  = ii_val, CYP = cyp_val,
      AUC_observed = auc_obs, 
      auc_ipred = as.numeric(auc_ipred),
      Cl = ind_params[1],
      Vc = ind_params[2],
      Q = ind_params[3],
      Vp = ind_params[4],
      Ktr = ind_params[5]
    )
    # --- MODIFICATION END ---
  )
}

# --- Prepare output file paths ---
stamp    <- format(Sys.Date(), "%Y%m%d")
pdf_file <- file.path(args$output_dir, paste0("tacro_mapbayest_plots_", stamp, ".pdf"))
csv_file <- file.path(args$output_dir, paste0("tacro_mapbayest_auc_",   stamp, ".csv"))

# --- Execute the analysis loop over all unique IDs ---
id_list <- obs_data_to_process %>% count(ID) %>% filter(n == 3) %>% pull(ID)

analysis_output <- future_map(id_list, ~run_one_id(.x, all_raw_data = raw_data, obs_data = obs_data_to_process), .progress = TRUE)

# --- Process and Save Results ---
# Separate the plots and the results data frames
all_plots   <- map(analysis_output, "plot")
results_df  <- map_dfr(analysis_output, "results")
# Open the PDF device to save plots
pdf(pdf_file, width = 8, height = 6)

# Save all the generated plots to a single PDF
message("Saving plots to PDF...")
pdf(pdf_file, width = 8, height = 6)
walk(all_plots, print) # Use walk() to print each plot in the list
dev.off()

write_csv(results_df, csv_file)
message(sprintf("✓ Patient plots saved to: %s", pdf_file))
message(sprintf("✓ AUC results saved to: %s", csv_file))


# -----------------------------------------------------------------------------
# 5. RESULTS: CALCULATE AND DISPLAY BIAS AND RMSE
# -----------------------------------------------------------------------------
message("Calculating final bias and RMSE...")
summary_stats <- results_df %>%
  mutate(
    bias = (auc_ipred - AUC_observed) / AUC_observed,
    bias_sq = bias^2
  ) %>%
  summarise(
    relative_bias_percent = mean(bias, na.rm = TRUE) * 100,
    rmse_percent = sqrt(mean(bias_sq, na.rm = TRUE)) * 100
  )

cat("\n--- Analysis Summary ---\n"); print(summary_stats); cat("------------------------\n\n")