# Title: Tacrolimus Pharmacokinetic Analysis 
# Author: benjamin maurel
# Date: 2025-09-09

suppressPackageStartupMessages(library(argparse))
suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(mrgsolve))
suppressPackageStartupMessages(library(mapbayr))
suppressPackageStartupMessages(library(MESS))
suppressPackageStartupMessages(library(lixoftConnectors))
library(glue)
suppressPackageStartupMessages(library(furrr))

parser <- ArgumentParser(description = "Run mapbayr analysis for Tacrolimus PK data.")
parser$add_argument("--virtual_cohort", type = "character", required = TRUE)
parser$add_argument("--output_dir", type = "character", default = ".")
parser$add_argument("--cores", type = "integer", default = 1)
parser$add_argument("--experiment", type = "character", default = ".")
parser$add_argument("--monolix_path", type = "character", default = NULL)
args <- parser$parse_args()

monolix_path <- args$monolix_path
if (is.null(monolix_path) || monolix_path == "") {
  monolix_path <- "/Applications/MonolixSuite2024R1.app/Contents/Resources/monolixSuite"
}

initializeLixoftConnectors(software = "monolix", path = monolix_path)
model_file_path   <- "test_model.txt"
data_file_path    <- file.path(args$output_dir, "virtual_cohort_train.csv")
project_save_path <- file.path(args$output_dir, "2_test_tacro.mlxtran")

column_mapping <- c(
  ID          = "ID",
  TIME        = "TIME",
  DV          = "DV",
  AMT         = "AMT",
  CYP         = "CYP",
  ST          = "ST",
  HT          = "HT"
)

newProject(
  modelFile = model_file_path,
  data = list(dataFile = data_file_path, headerTypes = column_mapping )
)

setIndividualParameterModel(list( correlationBlocks = list(id = list()), 
                                  covariateModel = list(CL = c(CYP = TRUE))) )
setIndividualParameterModel(list( correlationBlocks = list(id = list()), 
                                  covariateModel = list(Vc = c(ST = TRUE))) )
setIndividualParameterModel(list( correlationBlocks = list(id = list()), 
                                  covariateModel = list(KTR = c(ST = TRUE))) )

saveProject(projectFile = project_save_path)
runPopulationParameterEstimation()

monolix_results <- getEstimatedPopulationParameters()
results_df <- data.frame(Parameter = names(monolix_results), Value = unname(monolix_results))

omega_Cl <- monolix_results[["omega_CL"]]^2
omega_Vc <- monolix_results[["omega_Vc"]]^2
omega_Q <- monolix_results[["omega_Q"]]^2
omega_Vp <- monol_results[["omega_Vp"]]^2 # Note: verify key names from Monolix
omega_KTR <- monolix_results[["omega_KTR"]]^2
prop <- monolix_results[["b"]]
add <- monolix_results[['a']]

code_tac <- glue("
[PROB]
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
{{omega_Cl}}
{{omega_Vc}}
{{omega_Q}}
{{omega_Vp}}
{{omega_KTR}}

[MAIN]
double CL_app = TVCL * pow(CYPCL, CYP) * exp(ETA1 + ETA(1));
double V1_app = TVV1 * pow(STV1, ST) * exp(ETA2 + ETA(2));
double Q = TVQ * exp(ETA3 + ETA(3));
double V2 = TVV2 * exp(ETA4 + ETA(4));
double KTR = TVKTR * pow(STKTR, ST) * exp(ETA5 + ETA(5));

[SIGMA] @annotated
PROP : {{prop}}
ADD : {{add}}

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
")

mod_tac <- mcode("tac_model", code_tac)
mod_tac_updated <- param(
  mod_tac,
  TVCL  = monolix_results[["CL_pop"]],
  TVV1  = monolix_results[["Vc_pop"]],
  TVQ   = monolix_results[["Q_pop"]],
  TVV2  = monolix_results[["Vp_pop"]],
  TVKTR = monolix_results[["KTR_pop"]],
  CYPCL = exp(monolix_results[["beta_CL_CYP_1"]]),
  STKTR = exp(monolix_results[["beta_KTR_ST_1"]]),
  STV1  = exp(monolix_results[["beta_Vc_ST_1"]])
)

raw_data <- read_csv(args$virtual_cohort, na = c("null", ".", "NA", ""), trim_ws = TRUE, show_col_types = FALSE)
obs_data_to_process <- raw_data %>%
  filter(is.na(AMT)) %>%
  filter(near(TIME, 0) | between(TIME, 0.8, 1.2) | between(TIME, 2.4, 3.6))

if(args$cores > 1) { plan(multisession, workers = args$cores) } else { plan(sequential) }

run_one_id <- function(patient_id, all_raw_data, obs_data) {
  df_obs <- obs_data %>% filter(ID == patient_id)
  if(nrow(df_obs) != 3) return(NULL)
  
  df_dose <- all_raw_data %>% filter(ID == patient_id, !is.na(AMT), TIME == 0)
  if(nrow(df_dose) != 1) return(NULL)
  
  amt_val <- df_dose$AMT
  drug_val <- df_dose$DRUG
  cyp_val <- df_dose$CYP
  auc_obs <- df_dose$AUC
  st_val  <- if_else(drug_val == "Advagraf", 0, 1)
  ii_val  <- if_else(drug_val == "Advagraf", 24, 12)
  
  est_obj <- tryCatch({
    mod_tac_updated %>%
      adm_rows(time = 0, amt = amt_val, ss = 1, ii = ii_val, addl = 4) %>%
      add_covariates(CYP = cyp_val, ST = st_val) %>%
      obs_rows(time = df_obs$TIME[1], DV = df_obs$DV[1]) %>%
      obs_rows(time = df_obs$TIME[2], DV = df_obs$DV[2]) %>%
      obs_rows(time = df_obs$TIME[3], DV = df_obs$DV[3]) %>%
      mapbayest(verbose = FALSE)
  }, error = function(e) return(NULL))

  if (is.null(est_obj)) return(NULL)

  auc_start <- 0
  auc_end <- if (st_val == 1) 12 else 24
  aug <- mapbayr::augment(est_obj, start = auc_start, end = auc_end, delta = 0.1)
  
  ipred_win <- aug$aug_tab %>% filter(type == "IPRED", dplyr::between(time, 0, auc_end)) %>% transmute(time, DV = value)
  auc_ipred <- MESS::auc(ipred_win$time, ipred_win$DV)
  ind_params <- get_param(est_obj, .name = c("CL_app", "V1_app", "Q", "V2", "KTR"))
  
  list(
    plot = plot(aug, main = sprintf("ID %s — amt=%g mg — ii=%gh — ST=%s", patient_id, amt_val, ii_val, as.character(st_val)), xlim = c(0, 24)), 
    results = tibble(ID = patient_id, ST = st_val, amt = amt_val, ii = ii_val, CYP = cyp_val, AUC_observed = auc_obs, auc_ipred = as.numeric(auc_ipred), Cl = ind_params[1], Vc = ind_params[2], Q = ind_params[3], Vp = ind_params[4], Ktr = ind_params[5])
  )
}

stamp    <- format(Sys.Date(), "%Y%m%d")
pdf_file <- file.path(args$output_dir, paste0("tacro_mapbayest_plots_", stamp, ".pdf"))
csv_file <- file.path(args$output_dir, paste0("tacro_mapbayest_auc_",   stamp, ".csv"))

id_list <- obs_data_to_process %>% count(ID) %>% filter(n == 3) %>% pull(ID)
analysis_output <- future_map(id_list, ~run_one_id(.x, all_raw_data = raw_data, obs_data = obs_data_to_process), .options = furrr_options(seed = TRUE), .progress = TRUE)
analysis_output <- purrr::compact(analysis_output)
all_plots <- map(analysis_output, "plot")
results_df  <- map_dfr(analysis_output, "results")

pdf(pdf_file, width = 8, height = 6)
walk(all_plots, print) 
dev.off()
write_csv(results_df, csv_file)

summary_stats <- results_df %>% mutate(bias = (auc_ipred - AUC_observed) / AUC_observed, bias_sq = bias^2) %>% summarise(relative_bias_percent = mean(bias, na.rm = TRUE) * 100, rmse_percent = sqrt(mean(bias_sq, na.rm = TRUE)) * 100)
cat("\n--- Analysis Summary ---\n"); print(summary_stats); cat("------------------------\n\n")
