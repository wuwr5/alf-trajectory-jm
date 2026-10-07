# =====================================================================
# Bayesian shared-parameter joint models (JMbayes2) - acute liver failure ICU cohort
# Longitudinal marker trajectories + dynamic prediction of in-hospital death
#   + trajectory phenotype clustering
#
# Environment: R >= 4.4
# Input:  data/jm_wide.csv (id, tday, markers already log-transformed, covariates,
#                           event_time_d, event)
#         data/jm_surv.csv (one row per id: event_time_d, event, baseline covariates)
#
# Note: this script fits three *univariate* shared-parameter joint models
#       (bilirubin / INR / creatinine) and takes the union (row mean) of their Day-2
#       conditional risks. This uses all three longitudinal trajectories while avoiding
#       any mvglmer dependency. For the trivariate model with correlated random effects
#       see R/02_multivariate_jm.R (JMbayes2 supports list-of-lme natively).
#
# Dynamic prediction semantics (inside JMbayes2): in predict_Event,
#   last_times is taken from the survival event time in newdata (Surv(time)),
#   so the survival time in newdata MUST be reset to the landmark L (everyone in the risk
#   set is alive at L). The returned value is then the conditional cumulative incidence
#   P(L < T <= L+h | T > L, trajectory), i.e. the dynamic risk.
# =====================================================================
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
LOG    <- file.path(DATA, "jm_fit_log.txt")
log <- function(...) { cat(..., "\n"); cat(..., "\n", file = LOG, append = TRUE) }

wide <- read_csv_safe(file.path(DATA, "jm_wide.csv"))
surv_id <- read_csv_safe(file.path(DATA, "jm_surv.csv"))
log("Longitudinal wide table:", nrow(wide), "rows /", length(unique(wide$id)), "subjects")
log("Survival data:", nrow(surv_id), "subjects, events", sum(surv_id$event))

# Subject-level covariates (one row per id) + centring
ind <- surv_id[!duplicated(surv_id$id), ]
ind <- merge(ind, unique(wide[, c("id","MELD","SOFA")]), by = "id", all.x = TRUE)
ind$Age_c  <- (ind$Age - 60) / 10
ind$MELD_c <- (ind$MELD - 20) / 5
wide <- merge(wide, ind[, c("id","Age_c","MELD_c")], by = "id", all.x = TRUE)

# Shared-parameter survival submodel (Cox)
CoxFit <- coxph(Surv(event_time_d, event) ~ Age_c + gender_male + MELD_c,
                data = ind, model = TRUE)
log("Cox shared-parameter submodel fitted")

# ---------------------------------------------------------------- fit three univariate JMs
markers <- c("Bilirubin", "INR", "Creatinine")
fits   <- list(); ranefs <- list(); fit_ids <- list()
for (m in markers) {
  log("\n===== Fitting univariate JM:", m, "=====")
  fm <- lme(as.formula(paste0(m, " ~ ns(tday, 3)")),
            random = ~ ns(tday, 3) | id, data = wide,
            na.action = na.exclude,
            control = lmeControl(opt = "optim", msMaxIter = 200))
  jf <- tryCatch(
    jm(CoxFit, fm, time_var = "tday",
       n_chains = 3L, n_iter = 2000L, n_burnin = 1000L, n_thin = 2L,
       seed = 2024, cores = 3),
    error = function(e) { log("  !! jm(", m, ") failed: ", conditionMessage(e)); NULL })
  if (is.null(jf)) next
  fits[[m]] <- jf
  # Random-effects matrix (n_subj x q), rownames = subject id.
  # Use nlme's ranef(fm): its rownames ARE the subject ids and it matches the subject set
  # used by jm (these are exactly the b_hat values used for the JM conditioning).
  # Do NOT use jm$model_data$idL[[1]] (it contains NA cases dropped by lme).
  bmat <- as.matrix(ranef(fm))
  ranefs[[m]] <- bmat
  fit_ids[[m]] <- rownames(bmat)
  log("  ", m, " jm done; random-effects matrix", dim(bmat), "id examples:", head(rownames(bmat), 3))
}
saveRDS(fits, file.path(DATA, "jm_fits.rds"))
log("Saved jm_fits.rds (", length(fits), "models)")

# ---------------------------------------------------------------- dynamic prediction, landmark Day 2
L <- 2
# Day-2 landmark risk set (unified definition): exclude those with event == 1 who died
# within L days of ICU admission.
# Do NOT use event_time_d > L: that wrongly drops patients transferred out of the ICU within
# 2 days, with event == 0, who are still alive at Day 2.
risk <- ind[!((ind$event == 1) & (ind$event_time_d <= L)), ]
log("\n===== Dynamic prediction (landmark Day", L, ") risk set n =", nrow(risk), "=====")

pred_by_marker <- list()   # horizon -> marker -> data.frame(id, prob)
for (m in names(fits)) {
  for (h in c(7, 14)) {
    nd <- wide[wide$tday <= L & wide$id %in% risk$id,
               c("id","tday", m, "Age_c","MELD_c","gender_male","event_time_d","event")]
    nd <- nd[nd$id %in% fit_ids[[m]], ]          # keep only subjects fitted by this model
    nd$event_time_d <- L; nd$event <- 0           # reset the survival origin to the landmark
    p <- tryCatch(predict(fits[[m]], newdata = nd, process = "event",
                          times = L + h, return_newdata = TRUE),
                  error = function(e) { log("  !! predict", m, "h", h, "failed:", conditionMessage(e)); NULL })
    if (is.null(p)) next
    hor <- p[p$tday == L + h, ]                   # horizon row = dynamic risk
    out <- data.frame(id = hor$id, prob = hor$pred_CIF,
                      lo = hor$low_CIF, hi = hor$upp_CIF)
    pred_by_marker[[paste0(m,"_h",h)]] <- out
    log(sprintf("  %s h=%d: n=%d risk range %.3f..%.3f mean %.3f",
                m, h, nrow(out), min(out$prob), max(out$prob), mean(out$prob)))
  }
}

# Assemble one prediction table per horizon (per marker + union)
for (h in c(7, 14)) {
  base <- risk[, c("id","event_time_d","event")]
  tbl <- base
  avail <- c()
  for (m in names(fits)) {
    key <- paste0(m, "_h", h)
    if (!is.null(pred_by_marker[[key]])) {
      col <- pred_by_marker[[key]]
      col$prob[is.na(col$prob)] <- 0.5             # neutral value for missing predictions
      tbl <- merge(tbl, col[, c("id","prob","lo","hi")], by = "id", all.x = TRUE)
      names(tbl)[names(tbl) %in% c("prob","lo","hi")] <-
        c(paste0("prob_",m), paste0("lo_",m), paste0("hi_",m))
      avail <- c(avail, m)
    }
  }
  prob_cols <- paste0("prob_", avail)
  tbl$prob_JM <- rowMeans(tbl[, prob_cols, drop = FALSE])   # union dynamic risk
  tbl$y <- as.integer(tbl$event == 1 & tbl$event_time_d <= L + h)  # events within the horizon
  write.csv(tbl, file.path(DATA, sprintf("jm_pred_L%d_h%d.csv", L, h)), row.names = FALSE)
  ev <- sum(tbl$y); n <- nrow(tbl)
  log(sprintf("Wrote jm_pred_L%d_h%d.csv: n=%d events=%d (%.1f%%) union risk mean %.3f",
              L, h, n, ev, 100*ev/n, mean(tbl$prob_JM, na.rm=TRUE)))
}

# ---------------------------------------------------------------- trajectory phenotypes (random effects)
log("\n===== Trajectory phenotype clustering (subject-specific random effects b_hat) =====")
if (length(ranefs) >= 1) {
  # Combine per-marker random effects (intersection of ids) into a multi-dimensional summary
  common_ids <- Reduce(intersect, lapply(ranefs, rownames))
  b_all <- do.call(cbind, lapply(ranefs, function(x) x[common_ids, , drop = FALSE]))
  colnames(b_all) <- paste0(rep(names(ranefs), each = ncol(ranefs[[1]])), "_re", seq_len(ncol(ranefs[[1]])))
  b_sc <- scale(b_all)
  log("Combined random-effects matrix:", dim(b_all), " (shared ids:", length(common_ids), ")")

  png(file.path(DATA, "jm_elbow.png"), width = 1200, height = 800, res = 120)
  wss <- sapply(1:8, function(k) kmeans(b_sc, centers = k, nstart = 25, iter.max = 50)$tot.withinss)
  plot(1:8, wss, type = "b", xlab = "k", ylab = "WSS",
       main = "Elbow: number of trajectory phenotypes")
  dev.off()

  K <- 3
  set.seed(2024)
  cl <- kmeans(b_sc, centers = K, nstart = 50, iter.max = 100)
  clus <- data.frame(id = common_ids, cluster = cl$cluster)
  clus <- merge(clus, ind[, c("id","event_time_d","event")], by = "id")
  write.csv(clus, file.path(DATA, "jm_clusters.csv"), row.names = FALSE)
  log("Wrote jm_clusters.csv:", nrow(clus), "subjects, K =", K)
  log("Events per cluster (event):")
  print(with(clus, table(cluster, event)))
  log(capture.output(with(clus, table(cluster, event))))

  # Mean bilirubin trajectory per cluster (visualisation)
  sub <- merge(wide[, c("id","tday","Bilirubin")], clus[, "id", drop=FALSE], by="id")
  sub <- merge(sub, clus[, c("id","cluster")], by="id")
  png(file.path(DATA, "jm_cluster_traj.png"), width = 1400, height = 900, res = 120)
  cols <- c("steelblue","darkorange","seagreen","purple","brown")
  plot(NULL, xlim=c(0,14), ylim=c(min(wide$Bilirubin,na.rm=TRUE), max(wide$Bilirubin,na.rm=TRUE)),
       xlab="tday (days since ICU admission, log scale)", ylab="Bilirubin (log)",
       main="Mean bilirubin trajectory by phenotype")
  for (k in 1:K) {
    s <- sub[sub$cluster==k, ]
    mtraj <- aggregate(Bilirubin ~ tday, s, mean, na.rm=TRUE)
    lines(mtraj$tday, mtraj$Bilirubin, col=cols[k], lwd=2.5, type="b", pch=16)
  }
  legend("topright", legend=paste0("Cluster ",1:K), col=cols[1:K], lwd=2.5)
  dev.off()
  log("Wrote jm_cluster_traj.png")
}

log("\nDONE")
