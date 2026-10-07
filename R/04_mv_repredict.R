# 仅重做 JM-mv 动态预测与聚类(复用已保存的 jm_mv_fit.rds), 统一 Day2 风险集口径
# 口径: 排除 (event==1 & event_time_d<=L); 与 55/56/57 一致 -> 风险集应为 n=2331
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
LOG  <- file.path(DATA, "jm_mv_repred_log.txt")
log <- function(...) { cat(..., "\n"); cat(..., "\n", file = LOG, append = TRUE) }

wide <- read.csv(file.path(DATA, "jm_wide.csv"))
surv_id <- read.csv(file.path(DATA, "jm_surv.csv"))
ind <- surv_id[!duplicated(surv_id$id), ]
ind <- merge(ind, unique(wide[, c("id","MELD","SOFA")]), by = "id", all.x = TRUE)
ind$Age_c  <- (ind$Age - 60) / 10
ind$MELD_c <- (ind$MELD - 20) / 5
wide <- merge(wide, ind[, c("id","Age_c","MELD_c")], by = "id", all.x = TRUE)

jm_mv <- readRDS(file.path(DATA, "jm_mv_fit.rds"))
log("已加载 jm_mv_fit.rds")

L <- 2
# Day2 landmark 风险集(统一口径)
risk <- ind[!((ind$event == 1) & (ind$event_time_d <= L)), ]
log("\n===== JM-mv 动态预测 (landmark Day", L, ") 风险集 n =", nrow(risk), "=====")

for (h in c(7, 14)) {
  nd <- wide[wide$tday <= L & wide$id %in% risk$id,
             c("id","tday","Bilirubin","INR","Creatinine","Age_c","MELD_c","gender_male","event_time_d","event")]
  nd$event_time_d <- L; nd$event <- 0
  p <- tryCatch(predict(jm_mv, newdata = nd, process = "event",
                        times = L + h, return_newdata = TRUE),
                error = function(e) { log("  !! predict h", h, "失败:", conditionMessage(e)); NULL })
  if (is.null(p)) next
  hor <- p[p$tday == L + h, ]
  out <- data.frame(id = hor$id, prob_JMmv = hor$pred_CIF, lo = hor$low_CIF, hi = hor$upp_CIF)
  out <- merge(out, risk[, c("id","event_time_d","event")], by = "id")
  out$y <- as.integer(out$event == 1 & out$event_time_d <= L + h)
  write.csv(out, file.path(DATA, sprintf("jm_mv_pred_L%d_h%d.csv", L, h)), row.names = FALSE)
  n_miss <- nrow(risk) - nrow(out)
  log(sprintf("  jm_mv h=%d: 风险集=%d 预测出=%d 缺=%d 风险范围 %.3f..%.3f 均值 %.3f 事件=%d",
              h, nrow(risk), nrow(out), n_miss,
              min(out$prob_JMmv, na.rm=TRUE), max(out$prob_JMmv, na.rm=TRUE),
              mean(out$prob_JMmv, na.rm=TRUE), sum(out$y)))
}

# 追踪缺失个体(仅 h=7)
nd <- wide[wide$tday <= L & wide$id %in% risk$id,
           c("id","tday","Bilirubin","INR","Creatinine","Age_c","MELD_c","gender_male","event_time_d","event")]
nd$event_time_d <- L; nd$event <- 0
p7 <- predict(jm_mv, newdata = nd, process = "event", times = L + 7, return_newdata = TRUE)
got <- unique(p7$id[p7$tday == L + 7])
miss_ids <- setdiff(risk$id, got)
log(sprintf("\n未产出预测的 id 数: %d", length(miss_ids)))
if (length(miss_ids) > 0) {
  mi <- ind[ind$id %in% miss_ids, c("id","event_time_d","event","MELD","SOFA","Age")]
  nrec <- table(wide$id[wide$id %in% miss_ids])
  mi$n_long_records <- as.integer(nrec[as.character(mi$id)])
  log(capture.output(print(mi, row.names = FALSE)))
}

# 轨迹表型聚类: 直接取 jm 对象内个体随机效应后验均值 b (2501 x 12), 行序对应 model_data$idT 水平
log("\n===== JM-mv 轨迹表型聚类 =====")
b_all <- jm_mv$statistics$Mean$b
ids_b <- as.character(levels(jm_mv$model_data$idT))
stopifnot(nrow(b_all) == length(ids_b))
rownames(b_all) <- ids_b
b_sc <- scale(b_all)
log("随机效应矩阵:", dim(b_all))

png(file.path(DATA, "jm_mv_elbow.png"), width = 1200, height = 800, res = 120)
wss <- sapply(1:8, function(k) kmeans(b_sc, centers = k, nstart = 25, iter.max = 50)$tot.withinss)
plot(1:8, wss, type = "b", xlab = "k", ylab = "WSS", main = "Elbow: JM-mv 轨迹表型数")
dev.off()

K <- 3
set.seed(2024)
cl <- kmeans(b_sc, centers = K, nstart = 50, iter.max = 100)
clus <- data.frame(id = as.integer(ids_b), cluster = cl$cluster)
# 直接读原始 surv 表(含 case_id) 以保留 id 列类型一致
clus <- merge(clus, ind[, c("id","event_time_d","event")], by = "id")
write.csv(clus, file.path(DATA, "jm_mv_clusters.csv"), row.names = FALSE)
log("输出 jm_mv_clusters.csv:", nrow(clus), "例, K =", K)
log("各簇事件数 (event):"); log(capture.output(with(clus, table(cluster, event))))

sub <- merge(wide[, c("id","tday","Bilirubin")], clus[, c("id","cluster")], by = "id")
png(file.path(DATA, "jm_mv_cluster_traj.png"), width = 1400, height = 900, res = 120)
plot(NULL, xlim = c(0,14), ylim = c(min(wide$Bilirubin, na.rm=TRUE), max(wide$Bilirubin, na.rm=TRUE)),
     xlab = "tday (ICU 入科后天数, log 尺度)", ylab = "Bilirubin (log)", main = "JM-mv 各轨迹表型胆红素均值轨迹")
for (k in 1:K) {
  s <- sub[sub$cluster == k, ]; mtraj <- aggregate(Bilirubin ~ tday, s, mean, na.rm = TRUE)
  lines(mtraj$tday, mtraj$Bilirubin, col = c("steelblue","darkorange","seagreen")[k], lwd = 2.5, type = "b", pch = 16)
}
legend("topright", legend = paste0("Cluster ", 1:K), col = c("steelblue","darkorange","seagreen")[1:K], lwd = 2.5)
dev.off()
log("输出 jm_mv_cluster_traj.png")
log("\nJM_MV REPREDICT DONE")
