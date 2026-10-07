# Redo only the JM-mv dynamic prediction and clustering (reusing the saved jm_mv_fit.rds),
# with the unified Day-2 risk set definition.
# Definition: exclude (event == 1 & event_time_d <= L); consistent with scripts 05/06/07
# -> the risk set should be n = 2331.
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
LOG  <- file.path(DATA, "jm_mv_repred_log.txt")
log <- function(...) { cat(..., "\n"); cat(..., "\n", file = LOG, append = TRUE) }

wide <- read_csv_safe(file.path(DATA, "jm_wide.csv"))
surv_id <- read_csv_safe(file.path(DATA, "jm_surv.csv"))
ind <- surv_id[!duplicated(surv_id$id), ]
ind <- merge(ind, unique(wide[, c("id","MELD","SOFA")]), by = "id", all.x = TRUE)
ind$Age_c  <- (ind$Age - 60) / 10
ind$MELD_c <- (ind$MELD - 20) / 5
wide <- merge(wide, ind[, c("id","Age_c","MELD_c")], by = "id", all.x = TRUE)

jm_mv <- readRDS(file.path(DATA, "jm_mv_fit.rds"))
log("Loaded jm_mv_fit.rds")

L <- 2
# Day-2 landmark risk set (unified definition)
risk <- ind[!((ind$event == 1) & (ind$event_time_d <= L)), ]
log("\n===== JM-mv dynamic prediction (landmark Day", L, ") risk set n =", nrow(risk), "=====")

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
  n_miss <- nrow(risk) - nrow(out)
  log(sprintf("  jm_mv h=%d: risk set=%d predicted=%d missing=%d risk range %.3f..%.3f mean %.3f events=%d",
              h, nrow(risk), nrow(out), n_miss,
              min(out$prob_JMmv, na.rm=TRUE), max(out$prob_JMmv, na.rm=TRUE),
              mean(out$prob_JMmv, na.rm=TRUE), sum(out$y)))
}

# Trace the subjects with no prediction (h = 7 only)
nd <- wide[wide$tday <= L & wide$id %in% risk$id,
           c("id","tday","Bilirubin","INR","Creatinine","Age_c","MELD_c","gender_male","event_time_d","event")]
nd$event_time_d <- L; nd$event <- 0
p7 <- predict(jm_mv, newdata = nd, process = "event", times = L + 7, return_newdata = TRUE)
got <- unique(p7$id[p7$tday == L + 7])
miss_ids <- setdiff(risk$id, got)
log(sprintf("\nNumber of ids without a prediction: %d", length(miss_ids)))
if (length(miss_ids) > 0) {
  mi <- ind[ind$id %in% miss_ids, c("id","event_time_d","event","MELD","SOFA","Age")]
  nrec <- table(wide$id[wide$id %in% miss_ids])
  mi$n_long_records <- as.integer(nrec[as.character(mi$id)])
  log(capture.output(print(mi, row.names = FALSE)))
}

# Trajectory phenotype clustering: take the posterior means b of the subject-specific random
# effects from the jm object (2501 x 12); row order matches levels(model_data$idT).
log("\n===== JM-mv trajectory phenotype clustering =====")
b_all <- jm_mv$statistics$Mean$b
ids_b <- as.character(levels(jm_mv$model_data$idT))
stopifnot(nrow(b_all) == length(ids_b))
rownames(b_all) <- ids_b
b_sc <- scale(b_all)
log("Random-effects matrix:", dim(b_all))

png(file.path(DATA, "jm_mv_elbow.png"), width = 1200, height = 800, res = 120)
wss <- sapply(1:8, function(k) kmeans(b_sc, centers = k, nstart = 25, iter.max = 50)$tot.withinss)
plot(1:8, wss, type = "b", xlab = "k", ylab = "WSS",
     main = "Elbow: number of JM-mv trajectory phenotypes")
dev.off()

K <- 3
set.seed(2024)
cl <- kmeans(b_sc, centers = K, nstart = 50, iter.max = 100)
clus <- data.frame(id = as.integer(ids_b), cluster = cl$cluster)
# Read the raw survival table (which contains case_id) so that the id column type stays consistent
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
log("\nJM_MV REPREDICT DONE")
