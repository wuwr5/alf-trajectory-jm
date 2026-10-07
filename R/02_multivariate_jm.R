# 三变量共享参数联合模型 (JMbayes2 原生 list-of-lme) — 急性肝衰竭 ICU 队列
# 胆红素 + INR + 肌酐 三条纵向轨迹共享随机效应，landmark Day2 动态预测 + 轨迹聚类
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
DATA  <- Sys.getenv("ALF_DATA_DIR", unset = file.path(ROOT, "data"))
LOG    <- file.path(DATA, "jm_mv_fit_log.txt")
log <- function(...) { cat(..., "\n"); cat(..., "\n", file = LOG, append = TRUE) }

wide <- read.csv(file.path(DATA, "jm_wide.csv"))
surv_id <- read.csv(file.path(DATA, "jm_surv.csv"))
ind <- surv_id[!duplicated(surv_id$id), ]
ind <- merge(ind, unique(wide[, c("id","MELD","SOFA")]), by = "id", all.x = TRUE)
ind$Age_c  <- (ind$Age - 60) / 10
ind$MELD_c <- (ind$MELD - 20) / 5
wide <- merge(wide, ind[, c("id","Age_c","MELD_c")], by = "id", all.x = TRUE)
log("纵向宽表:", nrow(wide), "行 /", length(unique(wide$id)), "例")

CoxFit <- coxph(Surv(event_time_d, event) ~ Age_c + gender_male + MELD_c,
                data = ind, model = TRUE)
log("Cox 共享参数子模型拟合完成")

mk_lme <- function(m) {
  lme(as.formula(paste0(m, " ~ ns(tday,3)*Age_c + ns(tday,3)*MELD_c + ns(tday,3)*gender_male")),
      random = ~ ns(tday, 3) | id, data = wide, na.action = na.exclude,
      control = lmeControl(opt = "optim", msMaxIter = 200))
}
log("\n===== 拟合三变量联合模型 (Bilirubin+INR+Creatinine) =====")
fmB <- mk_lme("Bilirubin"); fmI <- mk_lme("INR"); fmC <- mk_lme("Creatinine")
jm_mv <- tryCatch(
  jm(CoxFit, list(fmB, fmI, fmC), time_var = "tday",
     n_chains = 3L, n_iter = 3000L, n_burnin = 1500L, n_thin = 2L, seed = 2024, cores = 3),
  error = function(e) { log("  !! jm_mv 失败: ", conditionMessage(e)); NULL })
if (is.null(jm_mv)) { log("JM_MV FAILED"); quit(status = 1) }
saveRDS(jm_mv, file.path(DATA, "jm_mv_fit.rds"))
log("三变量 JM 拟合完成; 关联参数可提取")

# 动态预测 landmark Day2
L <- 2
# Day2 landmark 风险集（统一口径）: 排除"入科后 <=L 天内已死亡(事件=1)"者;
# 不得用 event_time_d > L —— 那会误剔入科 <2 天转出ICU、event=0、Day2 仍存活的患者
risk <- ind[!((ind$event == 1) & (ind$event_time_d <= L)), ]
log("\n===== 三变量 JM 动态预测 (landmark Day", L, ") 风险集 n =", nrow(risk), "=====")
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
  log(sprintf("  jm_mv h=%d: n=%d 风险范围 %.3f..%.3f 均值 %.3f 事件=%d",
              h, nrow(out), min(out$prob_JMmv, na.rm=TRUE), max(out$prob_JMmv, na.rm=TRUE),
              mean(out$prob_JMmv, na.rm=TRUE), sum(out$y)))
}

# 轨迹表型聚类 (三变量随机效应, 12 维)
log("\n===== 三变量 JM 轨迹表型聚类 =====")
res <- lapply(list(fmB, fmI, fmC), function(f) as.matrix(ranef(f)))
common_ids <- Reduce(intersect, lapply(res, rownames))
b_all <- do.call(cbind, lapply(res, function(x) x[common_ids, , drop = FALSE]))
qn <- ncol(res[[1]])
colnames(b_all) <- paste0(rep(c("Bil","INR","Cre"), each = qn), "_re", seq_len(qn))
b_sc <- scale(b_all)
log("合并随机效应矩阵:", dim(b_all), " (共同 id:", length(common_ids), ")")

png(file.path(DATA, "jm_mv_elbow.png"), width = 1200, height = 800, res = 120)
wss <- sapply(1:8, function(k) kmeans(b_sc, centers = k, nstart = 25, iter.max = 50)$tot.withinss)
plot(1:8, wss, type = "b", xlab = "k", ylab = "WSS", main = "Elbow: JM-mv 轨迹表型数")
dev.off()

K <- 3
set.seed(2024)
cl <- kmeans(b_sc, centers = K, nstart = 50, iter.max = 100)
clus <- data.frame(id = common_ids, cluster = cl$cluster)
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
log("\nJM_MV DONE")
