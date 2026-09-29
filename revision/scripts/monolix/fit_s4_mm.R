# SAEM fit of the correctly specified scenario-4 population model in Monolix.
#
#   Rscript scripts/monolix/fit_s4_mm.R <data.csv> <out_dir> [threads] [variant]
#
# variant "true" (default): the generator's covariate model, as below.
# variant "l1": what a careful modeller would fit without knowing the simulator -- same
#   Michaelis-Menten structure, literature covariates (Ktr ~ ST, Vc ~ ST, Vmax ~ log(HT/35) + CYP),
#   but NO covariate on Km and NO (Km, Vc) correlation block (diagonal omega).
#
# Covariate model = the generator's: Ktr ~ ST, Vc ~ ST, Vmax ~ log(HT/35) + CYP,
# Km ~ log(HT/35); correlation block (Km, Vc); combined1 error (a + b*f), as simulated.
# Fixed-effect initial values = the published typical values the simulator is built on
# (what a pharmacometrician would start from); IIV, correlation and residual error start at
# Monolix defaults and are estimated.  Writes estimates.json + the .mlxtran project.
suppressPackageStartupMessages({ library(lixoftConnectors); library(jsonlite) })
a <- commandArgs(trailingOnly = TRUE)
data_csv <- normalizePath(a[1]); out_dir <- a[2]; threads <- if (length(a) >= 3) as.integer(a[3]) else 2
variant <- if (length(a) >= 4) a[4] else "true"
stopifnot(variant %in% c("true", "l1"))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE); out_dir <- normalizePath(out_dir)
model <- normalizePath("scripts/monolix/model_s4_mm.txt")
t0 <- Sys.time(); say <- function(...) cat(sprintf("[%s] ", format(Sys.time(), "%H:%M:%S")), ..., "\n", sep = "")

invisible(initializeLixoftConnectors(software = "monolix",
  path = "/Applications/monolixSuite2024R1.app/Contents/Resources/monolixSuite", force = TRUE))
setPreferences(threads = threads)
newProject(modelFile = model, data = list(dataFile = data_csv,
  headerTypes = c("id", "time", "observation", "amount", "catcov", "catcov", "contcov")))
addContinuousTransformedCovariate(tHT = "log(HT/35)")
if (variant == "true") {
  setCovariateModel(Ktr = c(ST = TRUE), Vc = c(ST = TRUE), Vmax = c(tHT = TRUE, CYP = TRUE), Km = c(tHT = TRUE))
  setCorrelationBlocks(id = list(c("Km", "Vc")))
} else {
  setCovariateModel(Ktr = c(ST = TRUE), Vc = c(ST = TRUE), Vmax = c(tHT = TRUE, CYP = TRUE))
}
# the observation model is named after the DATA column (DV), not the model output (CONC)
obs_name <- names(getContinuousObservationModel()$errorModel)[1]
do.call(setErrorModel, setNames(list("combined1"), obs_name))
setPopulationParameterInformation(
  Ktr_pop  = list(initialValue = 3.34),  Vc_pop  = list(initialValue = 486),
  Q_pop    = list(initialValue = 79),    Vp_pop  = list(initialValue = 271),
  Vmax_pop = list(initialValue = 0.212), Km_pop  = list(initialValue = 0.01),
  beta_Ktr_ST_1 = list(initialValue = log(1.53)), beta_Vc_ST_1 = list(initialValue = log(0.29)),
  beta_Vmax_tHT = list(initialValue = -3.14), beta_Vmax_CYP_1 = list(initialValue = log(2)))
if (variant == "true") setPopulationParameterInformation(beta_Km_tHT = list(initialValue = 1.0))
# combined1 is a + b*f; an exponent c, if listed, must stay at 1 or the error model changes
pinfo <- getPopulationParameterInformation()
if ("c" %in% pinfo$name) setPopulationParameterInformation(c = list(initialValue = 1, method = "FIXED"))
pinfo <- getPopulationParameterInformation()
print(pinfo[, intersect(c("name", "initialValue", "method"), names(pinfo))])
cat("error model:", getContinuousObservationModel()$formula, "\n")
proj <- file.path(out_dir, "s4_mm.mlxtran"); saveProject(projectFile = proj)
say("project saved: ", proj, " | ", threads, " threads | variant ", variant)
say("population parameters to estimate: ", paste(getPopulationParameterInformation()$name, collapse = ", "))
runPopulationParameterEstimation()
est <- getEstimatedPopulationParameters()
say("SAEM done after ", round(as.numeric(Sys.time() - t0, units = "mins"), 1), " min")
print(round(est, 5))
runStandardErrorEstimation(linearization = TRUE)
se <- tryCatch(getEstimatedStandardErrors()$linearization, error = function(e) NULL)
write_json(list(data = data_csv, variant = variant, n_patients = length(unique(read.csv(data_csv)$ID)),
                estimates = as.list(est), se = if (is.null(se)) NULL else setNames(as.list(se$se), se$parameter),
                minutes = as.numeric(Sys.time() - t0, units = "mins")),
           file.path(out_dir, "estimates.json"), auto_unbox = TRUE, digits = 10, pretty = TRUE)
say("wrote ", file.path(out_dir, "estimates.json"))
cat("FIT_DONE\n")
