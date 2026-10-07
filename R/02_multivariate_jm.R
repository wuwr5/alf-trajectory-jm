# Trivariate shared-parameter joint model (JMbayes2 native list-of-lme) - ALF ICU cohort
# Bilirubin + INR + creatinine share random effects; landmark Day-2 dynamic prediction
#   + trajectory clustering
suppressPackageStartupMessages({
  library(JMbayes2); library(splines); library(survival); library(nlme)
})
# --- Path resolution: overridable via environment variables, defaults to data/ in the repo ---
.this <- tryCatch({
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) normalizePath(f, winslash = "/", mustWork = FALSE) else NA_character_
}, error = function(e) NA_character_)
ROOT <- Sys.getenv("ALF_ROOT", unset = "")
if (!nzchar(ROOT)) {
  ROOT <- if (!is.na(.this)) dirname(dirname(.this)) else getwd()
}
DATA <- Sys.getenv("ALF_DATA_DIR", unset = file.path(ROOT, "data"))
dir.create(DATA, recursive = TRUE, showWarnings = FALSE)

# --- BOM-tolerant CSV reader -------------------------------------------------
# pandas writes a UTF-8 BOM when encoding="utf-8-sig" is used. R's read.csv then
# names the first column "X...id" instead of "id", silently breaking every join.
# read_csv_safe() strips a leading BOM so both BOM and BOM-free files work.
read_csv_safe <- function(path, ...) {
  read.csv(path, fileEncoding = "UTF-8-BOM", ...)
}


set.seed(2024)
LOG    <- file.path(DATA, "jm_mv_fit_log.txt")
log <- function(...) { cat(..., "\n"); cat(..., "\n", file = LOG, append = TRUE) }

wide <- read_csv_safe(file.path(DATA, "jm_wide.csv"))
surv_id <- read_csv_safe(file.path(DATA, "jm_surv.csv"))
ind <- surv_id[!duplicated(surv_id$id), ]
ind <- merge(ind, unique(wide[, c("id","MELD","SOFA")]), by = "id", all.x = TRUE)
ind$Age_c  <- (ind$Age - 60) / 10
ind$MELD_c <- (ind$MELD - 20) / 5
wide <- merge(wide, ind[, c("id","Age_c","MELD_c")], by = "id", all.x = TRUE)
log("Longitudinal wide table:", nrow(wide), "rows /", length(unique(wide$id)), "subjects")

CoxFit <- coxph(Surv(event_time_d, event) ~ Age_c + gender_male + MELD_c,
                data = ind, model = TRUE)
log("Cox shared-parameter submodel fitted")

mk_lme <- function(m) {
  lme(as.formula(paste0(m, " ~ ns(tday,3)*Age_c + ns(tday,3)*MELD_c + ns(tday,3)*gender_male")),
      random = ~ ns(tday, 3) | id, data = wide, na.action = na.exclude,
      control = lmeControl(opt = "optim", msMaxIter = 200))
}
log("\n===== Fitting trivariate joint model (Bilirubin + INR + Creatinine) =====")
fmB <- mk_lme("Bilirubin"); fmI <- mk_lme("INR"); fmC <- mk_lme("Creatinine")
jm_mv <- tryCatch(
  jm(CoxFit, list(fmB, fmI, fmC), time_var = "tday",
     n_chains = 3L, n_iter = 3000L, n_burnin = 1500L, n_thin = 2L, seed = 2024, cores = 3),
  error = function(e) { log("  !! jm_mv failed: ", conditionMessage(e)); NULL })
if (is.null(jm_mv)) { log("JM_MV FAILED"); quit(status = 1) }
saveRDS(jm_mv, file.path(DATA, "jm_mv_fit.rds"))
log("Trivariate JM fitted; association parameters available")

# Dynamic prediction, landmark Day 2
L <- 2
# Day-2 landmark risk set (unified definition): exclude those with event == 1 who died within
# L days. Do NOT use event_time_d > L: that wrongly drops patients transferred out of the ICU
# within 2 days, with event == 0, who are still alive at Day 2.
risk <- ind[!((ind$event == 1) & (ind$event_time_d <= L)), ]
log("\n===== Trivariate JM dynamic prediction (landmark Day", L, ") risk set n =", nrow(risk), "=====")
for (h in c(7, 14)) {
  nd <- wide[wide$tday <= L & wide$id %in% risk$id,
             c("id","tday","Bilirubin","INR","Creatinine","Age_c","MELD_c","gender_male","event_time_d","event")]
  nd$event_time_d <- L; nd$event <- 0
  p <- tryCatch(predict(jm_mv, newdata = nd, process = "event",
                        times = L + h, return_newdata = TRUE),
                error = function(e) { log("  !! predict h", h, "failed:", conditionMessage(e)); NULL })
  if (is.null(p)) next
  hor <- p[p$tday == L + h, ]
  out <- data.frame(id = hor$id, prob_JMmv = hor$pred_CIF, lo = hor$low_CIF, hi = hor$upp_CIF)
  out <- merge(out, risk[, c("id","event_time_d","event")], by = "id")
  out$y <- as.integer(out$event == 1 & out$event_time_d <= L + h)
  write.csv(out, file.path(DATA, sprintf("jm_mv_pred_L%d_h%d.csv", L, h)), row.names = FALSE)
  log(sprintf("  jm_mv h=%d: n=%d risk range %.3f..%.3f mean %.3f events=%d",
              h, nrow(out), min(out$prob_JMmv, na.rm=TRUE), max(out$prob_JMmv, na.rm=TRUE),
              mean(out$prob_JMmv, na.rm=TRUE), sum(out$y)))
}

# Trajectory phenotype clustering (trivariate random effects, 12 dimensions)
log("\n===== Trivariate JM trajectory phenotype clustering =====")
res <- lapply(list(fmB, fmI, fmC), function(f) as.matrix(ranef(f)))
common_ids <- Reduce(intersect, lapply(res, rownames))
b_all <- do.call(cbind, lapply(res, function(x) x[common_ids, , drop = FALSE]))
qn <- ncol(res[[1]])
colnames(b_all) <- paste0(rep(c("Bil","INR","Cre"), each = qn), "_re", seq_len(qn))
b_sc <- scale(b_all)
log("Combined random-effects matrix:", dim(b_all), " (shared ids:", length(common_ids), ")")

png(file.path(DATA, "jm_mv_elbow.png"), width = 1200, height = 800, res = 120)
wss <- sapply(1:8, function(k) kmeans(b_sc, centers = k, nstart = 25, iter.max = 50)$tot.withinss)
plot(1:8, wss, type = "b", xlab = "k", ylab = "WSS",
     main = "Elbow: number of JM-mv trajectory phenotypes")
dev.off()

K <- 3
set.seed(2024)
cl <- kmeans(b_sc, centers = K, nstart = 50, iter.max = 100)
clus <- data.frame(id = common_ids, cluster = cl$cluster)
clus <- merge(clus, ind[, c("id","event_time_d","event")], by = "id")
write.csv(clus, file.path(DATA, "jm_mv_clusters.csv"), row.names = FALSE)
log("Wrote jm_mv_clusters.csv:", nrow(clus), "subjects, K =", K)
log("Events per cluster (event):"); log(capture.output(with(clus, table(cluster, event))))

sub <- merge(wide[, c("id","tday","Bilirubin")], clus[, c("id","cluster")], by = "id")
png(file.path(DATA, "jm_mv_cluster_traj.png"), width = 1400, height = 900, res = 120)
plot(NULL, xlim = c(0,14), ylim = c(min(wide$Bilirubin, na.rm=TRUE), max(wide$Bilirubin, na.rm=TRUE)),
     xlab = "tday (days since ICU admission, log scale)", ylab = "Bilirubin (log)",
     main = "JM-mv mean bilirubin trajectory by phenotype")
for (k in 1:K) {
  s <- sub[sub$cluster == k, ]; mtraj <- aggregate(Bilirubin ~ tday, s, mean, na.rm = TRUE)
  lines(mtraj$tday, mtraj$Bilirubin, col = c("steelblue","darkorange","seagreen")[k], lwd = 2.5, type = "b", pch = 16)
}
legend("topright", legend = paste0("Cluster ", 1:K), col = c("steelblue","darkorange","seagreen")[1:K], lwd = 2.5)
dev.off()
log("Wrote jm_mv_cluster_traj.png")
log("\nJM_MV DONE")
