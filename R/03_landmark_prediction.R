# 仅重做 JM 单变量臂(胆红素/INR/肌酐 并集)动态预测, 复用 jm_fits.rds, 统一 Day2 风险集口径
# 口径: 排除 (event==1 & event_time_d<=L) -> 风险集应为 n=2331
suppressPackageStartupMessages({
  library(JMbayes2); library(splines); library(survival); library(nlme)
})
# --- 路径解析：支持环境变量覆盖，默认仓库内 data/ ---
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

set.seed(2024)
DATA <- Sys.getenv("ALF_DATA_DIR", unset = file.path(ROOT, "data"))
LOG  <- file.path(DATA, "jm_repred_log.txt")
log <- function(...) { cat(..., "\n"); cat(..., "\n", file = LOG, append = TRUE) }

wide <- read.csv(file.path(DATA, "jm_wide.csv"))
surv_id <- read.csv(file.path(DATA, "jm_surv.csv"))
ind <- surv_id[!duplicated(surv_id$id), ]
ind <- merge(ind, unique(wide[, c("id","MELD","SOFA")]), by = "id", all.x = TRUE)
ind$Age_c  <- (ind$Age - 60) / 10
ind$MELD_c <- (ind$MELD - 20) / 5
wide <- merge(wide, ind[, c("id","Age_c","MELD_c")], by = "id", all.x = TRUE)

fits <- readRDS(file.path(DATA, "jm_fits.rds"))
log("已加载 jm_fits.rds: ", paste(names(fits), collapse=", "))
# jm 对象无 ranef 方法; 受试者集取 model_data$idT 的水平(即进入 Cox 部分的个体)
fit_ids <- lapply(fits, function(f) as.character(levels(f$model_data$idT)))
log("各 marker 拟合受试者数: ", paste(names(fits), sapply(fit_ids, length), sep="=", collapse=", "))

L <- 2
risk <- ind[!((ind$event == 1) & (ind$event_time_d <= L)), ]
log("\n===== JM 单变量臂动态预测 (landmark Day", L, ") 风险集 n =", nrow(risk), "=====")

pred_by_marker <- list()
for (m in names(fits)) {
  for (h in c(7, 14)) {
    nd <- wide[wide$tday <= L & as.character(wide$id) %in% as.character(risk$id), ]
    nd <- nd[as.character(nd$id) %in% fit_ids[[m]], ]
    nd$event_time_d <- L; nd$event <- 0
    p <- tryCatch(predict(fits[[m]], newdata = nd, process = "event",
                          times = L + h, return_newdata = TRUE),
                  error = function(e) { log("  !! predict", m, "h", h, "失败:", conditionMessage(e)); NULL })
    if (is.null(p)) next
    hor <- p[p$tday == L + h, ]
    out <- data.frame(id = as.character(hor$id), prob = hor$pred_CIF, lo = hor$low_CIF, hi = hor$upp_CIF)
    pred_by_marker[[paste0(m,"_h",h)]] <- out
    log(sprintf("  %s h=%d: n=%d 风险范围 %.3f..%.3f 均值 %.3f",
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
  log(sprintf("输出 jm_pred_L%d_h%d.csv: n=%d 事件=%d (%.1f%%) 并集风险均值 %.3f",
              L, h, n, ev, 100*ev/n, mean(tbl$prob_JM, na.rm=TRUE)))
}
log("\nJM REPREDICT DONE")
