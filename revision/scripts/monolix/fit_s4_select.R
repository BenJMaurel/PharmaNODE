# Structural model selection for the hidden-saturation experiment (handoff 11): fit ONE candidate
# structure to a training cohort with SAEM, then compute the log-likelihood by importance sampling
# and report -2LL, AIC, BIC, BICc.
#
#   Rscript scripts/monolix/fit_s4_select.R <data.csv> <out_dir> <threads> <lin|mm> [k_init]
#
# Both candidates use the covariate model a careful modeller would fit without knowing the
# simulator (as variant "l1" of fit_s4_mm.R): Ktr ~ ST, Vc ~ ST, elimination ~ log(HT/35) + CYP,
# diagonal omega, combined1 error (a + b*f).
#   lin: CL   ~ log(HT/35) + CYP, starting from the published typical values (CL 21.2 L/h).
#   mm : Vmax ~ log(HT/35) + CYP, no covariate on Km, started at the TRUE scaled values
#        (Vmax 0.212*k, Km 0.01*k): Michaelis-Menten gets its best shot, so a tie is not caused by
#        a poorly started MM fit.
# Writes estimates.json (estimates, s.e., log-likelihood block) + the .mlxtran project.
suppressPackageStartupMessages({ library(lixoftConnectors); library(jsonlite) })
a <- commandArgs(trailingOnly = TRUE)
data_csv <- normalizePath(a[1]); out_dir <- a[2]; threads <- as.integer(a[3]); struct <- a[4]
k_init <- if (length(a) >= 5) as.numeric(a[5]) else 1
stopifnot(struct %in% c("lin", "mm"))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE); out_dir <- normalizePath(out_dir)
model <- normalizePath(sprintf("scripts/monolix/model_s4_%s.txt", struct))
t0 <- Sys.time(); say <- function(...) cat(sprintf("[%s] ", format(Sys.time(), "%H:%M:%S")), ..., "\n", sep = "")

invisible(initializeLixoftConnectors(software = "monolix",
  path = "/Applications/monolixSuite2024R1.app/Contents/Resources/monolixSuite", force = TRUE))
setPreferences(threads = threads)
newProject(modelFile = model, data = list(dataFile = data_csv,
  headerTypes = c("id", "time", "observation", "amount", "catcov", "catcov", "contcov")))
addContinuousTransformedCovariate(tHT = "log(HT/35)")
if (struct == "lin") {
  setCovariateModel(Ktr = c(ST = TRUE), Vc = c(ST = TRUE), CL = c(tHT = TRUE, CYP = TRUE))
} else {
  setCovariateModel(Ktr = c(ST = TRUE), Vc = c(ST = TRUE), Vmax = c(tHT = TRUE, CYP = TRUE))
}
obs_name <- names(getContinuousObservationModel()$errorModel)[1]
do.call(setErrorModel, setNames(list("combined1"), obs_name))
setPopulationParameterInformation(
  Ktr_pop = list(initialValue = 3.34), Vc_pop = list(initialValue = 486),
  Q_pop   = list(initialValue = 79),   Vp_pop = list(initialValue = 271),
  beta_Ktr_ST_1 = list(initialValue = log(1.53)), beta_Vc_ST_1 = list(initialValue = log(0.29)))
if (struct == "lin") {
  setPopulationParameterInformation(CL_pop = list(initialValue = 21.2),
    beta_CL_tHT = list(initialValue = -3.14), beta_CL_CYP_1 = list(initialValue = log(2)))
} else {
  setPopulationParameterInformation(Vmax_pop = list(initialValue = 0.212 * k_init),
    Km_pop = list(initialValue = 0.01 * k_init),
    beta_Vmax_tHT = list(initialValue = -3.14), beta_Vmax_CYP_1 = list(initialValue = log(2)))
}
pinfo <- getPopulationParameterInformation()
if ("c" %in% pinfo$name) setPopulationParameterInformation(c = list(initialValue = 1, method = "FIXED"))
pinfo <- getPopulationParameterInformation()
print(pinfo[, intersect(c("name", "initialValue", "method"), names(pinfo))])
proj <- file.path(out_dir, sprintf("s4_%s.mlxtran", struct)); saveProject(projectFile = proj)
say("project saved: ", proj, " | ", threads, " threads | struct ", struct, " | k_init ", k_init)
runPopulationParameterEstimation()
est <- getEstimatedPopulationParameters()
say("SAEM done after ", round(as.numeric(Sys.time() - t0, units = "mins"), 1), " min")
print(round(est, 5))
runConditionalDistributionSampling()
runLogLikelihoodEstimation(linearization = FALSE)
ll <- getEstimatedLogLikelihood()
say("log-likelihood (importance sampling):"); print(ll)
runStandardErrorEstimation(linearization = TRUE)
se <- tryCatch(getEstimatedStandardErrors()$linearization, error = function(e) NULL)
saveProject(projectFile = proj)
write_json(list(data = data_csv, struct = struct, k_init = k_init,
                n_patients = length(unique(read.csv(data_csv)$ID)),
                n_obs = sum(read.csv(data_csv)$DV != "."),
                estimates = as.list(est), loglik = ll,
                se = if (is.null(se)) NULL else setNames(as.list(se$se), se$parameter),
                minutes = as.numeric(Sys.time() - t0, units = "mins")),
           file.path(out_dir, "estimates.json"), auto_unbox = TRUE, digits = 10, pretty = TRUE)
say("wrote ", file.path(out_dir, "estimates.json"))
cat("FIT_DONE\n")
