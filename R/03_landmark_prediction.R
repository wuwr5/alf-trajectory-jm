# Redo only the dynamic prediction for the univariate JM arm (bilirubin / INR / creatinine union),
# reusing jm_fits.rds, with the unified Day-2 risk set definition.
# Definition: exclude (event == 1 & event_time_d <= L) -> the risk set should be n = 2331.
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
LOG  <- file.path(DATA, "jm_repred_log.txt")
log <- function(...) { cat(..., "\n"); cat(..., "\n", file = LOG, append = TRUE) }

wide <- read_csv_safe(file.path(DATA, "jm_wide.csv"))
surv_id <- read_csv_safe(file.path(DATA, "jm_surv.csv"))
ind <- surv_id[!duplicated(surv_id$id), ]
ind <- merge(ind, unique(wide[, c("id","MELD","SOFA")]), by = "id", all.x = TRUE)
ind$Age_c  <- (ind$Age - 60) / 10
ind$MELD_c <- (ind$MELD - 20) / 5
wide <- merge(wide, ind[, c("id","Age_c","MELD_c")], by = "id", all.x = TRUE)

fits <- readRDS(file.path(DATA, "jm_fits.rds"))
log("Loaded jm_fits.rds: ", paste(names(fits), collapse=", "))
# jm objects have no ranef method; take the subject set from the levels of model_data$idT
# (i.e. the individuals entering the Cox part).
fit_ids <- lapply(fits, function(f) as.character(levels(f$model_data$idT)))
log("Subjects fitted per marker: ", paste(names(fits), sapply(fit_ids, length), sep="=", collapse=", "))

L <- 2
risk <- ind[!((ind$event == 1) & (ind$event_time_d <= L)), ]
log("\n===== Univariate JM arm dynamic prediction (landmark Day", L, ") risk set n =", nrow(risk), "=====")

pred_by_marker <- list()
for (m in names(fits)) {
  for (h in c(7, 14)) {
    nd <- wide[wide$tday <= L & as.character(wide$id) %in% as.character(risk$id), ]
    nd <- nd[as.character(nd$id) %in% fit_ids[[m]], ]
    nd$event_time_d <- L; nd$event <- 0
    p <- tryCatch(predict(fits[[m]], newdata = nd, process = "event",
                          times = L + h, return_newdata = TRUE),
                  error = function(e) { log("  !! predict", m, "h", h, "failed:", conditionMessage(e)); NULL })
    if (is.null(p)) next
    hor <- p[p$tday == L + h, ]
    out <- data.frame(id = as.character(hor$id), prob = hor$pred_CIF, lo = hor$low_CIF, hi = hor$upp_CIF)
    pred_by_marker[[paste0(m,"_h",h)]] <- out
    log(sprintf("  %s h=%d: n=%d risk range %.3f..%.3f mean %.3f",
                m, h, nrow(out), min(out$prob), max(out$prob), mean(out$prob)))
  }
}

for (h in c(7, 14)) {
  base <- risk[, c("id","event_time_d","event")]
  base$id <- as.character(base$id)
  tbl <- base
  avail <- c()
  for (m in names(fits)) {
    key <- paste0(m, "_h", h)
    if (!is.null(pred_by_marker[[key]])) {
      col <- pred_by_marker[[key]]
      col$prob[is.na(col$prob)] <- 0.5
      tbl <- merge(tbl, col[, c("id","prob","lo","hi")], by = "id", all.x = TRUE)
      names(tbl)[names(tbl) %in% c("prob","lo","hi")] <-
        c(paste0("prob_",m), paste0("lo_",m), paste0("hi_",m))
      avail <- c(avail, m)
    }
  }
  prob_cols <- paste0("prob_", avail)
  tbl$prob_JM <- rowMeans(tbl[, prob_cols, drop = FALSE])
  tbl$y <- as.integer(tbl$event == 1 & tbl$event_time_d <= L + h)
  write.csv(tbl, file.path(DATA, sprintf("jm_pred_L%d_h%d.csv", L, h)), row.names = FALSE)
  ev <- sum(tbl$y); n <- nrow(tbl)
  log(sprintf("Wrote jm_pred_L%d_h%d.csv: n=%d events=%d (%.1f%%) union risk mean %.3f",
              L, h, n, ev, 100*ev/n, mean(tbl$prob_JM, na.rm=TRUE)))
}
log("\nJM REPREDICT DONE")
