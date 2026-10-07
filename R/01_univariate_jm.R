# =====================================================================
# 贝叶斯共享参数联合模型（JMbayes2）— 急性肝衰竭 ICU 队列
# 纵向标志物轨迹 + 院内死亡动态预测 + 轨迹表型聚类
#
# 环境: R-4.4.2 (D:/analysis_software/R-4.4.2)
# 输入: data/jm_wide.csv (id, tday, 各 marker 已 log 变换, 协变量, event_time_d, event)
#       data/jm_surv.csv (id 唯一: event_time_d, event, 基线协变量)
#
# 重要: GLMMadaptive 0.9-0 起移除了 mvglmer，故 TRUE 多变量共享参数
#       (相关随机效应) 暂不可用。本脚本改用 **三个单变量共享参数 JM**
#       (胆红素 / INR / 肌酐) 的并集预测，等价于用全部三条纵向轨迹，
#       避免 mvglmer 依赖；多变量相关随机效应版需 GLMMadaptive < 0.9 (见报告)。
#
# 动态预测语义 (JMbayes2 内部): predict_Event 中
#   last_times = newdata 的生存事件时间 (Surv(time)),
#   故必须把 newdata 的生存时间重置为 landmark L(风险集在 L 时均存活),
#   预测返回条件累积发生率 P(L < T <= L+h | T > L, 轨迹) = 动态风险。
# =====================================================================
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
LOG    <- file.path(DATA, "jm_fit_log.txt")
log <- function(...) { cat(..., "\n"); cat(..., "\n", file = LOG, append = TRUE) }

wide <- read.csv(file.path(DATA, "jm_wide.csv"))
surv_id <- read.csv(file.path(DATA, "jm_surv.csv"))
log("纵向宽表:", nrow(wide), "行 /", length(unique(wide$id)), "例")
log("生存数据:", nrow(surv_id), "例，事件", sum(surv_id$event))

# 个体级协变量(每 id 一行) + 中心化
ind <- surv_id[!duplicated(surv_id$id), ]
ind <- merge(ind, unique(wide[, c("id","MELD","SOFA")]), by = "id", all.x = TRUE)
ind$Age_c  <- (ind$Age - 60) / 10
ind$MELD_c <- (ind$MELD - 20) / 5
wide <- merge(wide, ind[, c("id","Age_c","MELD_c")], by = "id", all.x = TRUE)

# 共享参数生存子模型 (Cox)
CoxFit <- coxph(Surv(event_time_d, event) ~ Age_c + gender_male + MELD_c,
                data = ind, model = TRUE)
log("Cox 共享参数子模型拟合完成")

# ---------------------------------------------------------------- 拟合三个单变量 JM
markers <- c("Bilirubin", "INR", "Creatinine")
fits   <- list(); ranefs <- list(); fit_ids <- list()
for (m in markers) {
  log("\n===== 拟合单变量 JM:", m, "=====")
  fm <- lme(as.formula(paste0(m, " ~ ns(tday, 3)")),
            random = ~ ns(tday, 3) | id, data = wide,
            na.action = na.exclude,
            control = lmeControl(opt = "optim", msMaxIter = 200))
  jf <- tryCatch(
    jm(CoxFit, fm, time_var = "tday",
       n_chains = 3L, n_iter = 2000L, n_burnin = 1000L, n_thin = 2L,
       seed = 2024, cores = 3),
    error = function(e) { log("  !! jm(", m, ") 失败: ", conditionMessage(e)); NULL })
  if (is.null(jf)) next
  fits[[m]] <- jf
  # 随机效应矩阵 (n_subj x q)，行名=受试者 id。
  # 注意: 用 nlme 的 ranef(fm) 行名(id) 对齐——它正是 JM 条件所用的 b_hat，
  # 与 jm 使用的受试者集合一致；勿用 jm$model_data$idL[[1]](含被 lme 剔除的 NA 个案)。
  bmat <- as.matrix(ranef(fm))
  ranefs[[m]] <- bmat
  fit_ids[[m]] <- rownames(bmat)
  log("  ", m, " jm 完成; 随机效应矩阵", dim(bmat), "行名示例:", head(rownames(bmat), 3))
}
saveRDS(fits, file.path(DATA, "jm_fits.rds"))
log("已保存 jm_fits.rds (", length(fits), "个模型)")

# ---------------------------------------------------------------- 动态预测 landmark Day2
L <- 2
# Day2 landmark 风险集(统一口径): 排除 "入科后 <=L 天内已死亡(event==1)" 者。
# 不得用 event_time_d > L —— 那会误剔入科 <2 天转出 ICU、event==0、Day2 仍存活的患者。
risk <- ind[!((ind$event == 1) & (ind$event_time_d <= L)), ]
log("\n===== 动态预测 (landmark Day", L, ") 风险集 n =", nrow(risk), "=====")

pred_by_marker <- list()   # horizon -> marker -> data.frame(id, prob)
for (m in names(fits)) {
  for (h in c(7, 14)) {
    nd <- wide[wide$tday <= L & wide$id %in% risk$id,
               c("id","tday", m, "Age_c","MELD_c","gender_male","event_time_d","event")]
    nd <- nd[nd$id %in% fit_ids[[m]], ]          # 仅保留本模型拟合过的受试者
    nd$event_time_d <- L; nd$event <- 0           # 重置生存原点到 landmark
    p <- tryCatch(predict(fits[[m]], newdata = nd, process = "event",
                          times = L + h, return_newdata = TRUE),
                  error = function(e) { log("  !! predict", m, "h", h, "失败:", conditionMessage(e)); NULL })
    if (is.null(p)) next
    hor <- p[p$tday == L + h, ]                   # horizon 行 = 动态风险
    out <- data.frame(id = hor$id, prob = hor$pred_CIF,
                      lo = hor$low_CIF, hi = hor$upp_CIF)
    pred_by_marker[[paste0(m,"_h",h)]] <- out
    log(sprintf("  %s h=%d: n=%d 风险范围 %.3f..%.3f 均值 %.3f",
                m, h, nrow(out), min(out$prob), max(out$prob), mean(out$prob)))
  }
}

# 汇总为每个 horizon 的预测表 (各 marker + 并集)
for (h in c(7, 14)) {
  base <- risk[, c("id","event_time_d","event")]
  tbl <- base
  avail <- c()
  for (m in names(fits)) {
    key <- paste0(m, "_h", h)
    if (!is.null(pred_by_marker[[key]])) {
      col <- pred_by_marker[[key]]
      col$prob[is.na(col$prob)] <- 0.5             # 缺失预测置中性
      tbl <- merge(tbl, col[, c("id","prob","lo","hi")], by = "id", all.x = TRUE)
      names(tbl)[names(tbl) %in% c("prob","lo","hi")] <-
        c(paste0("prob_",m), paste0("lo_",m), paste0("hi_",m))
      avail <- c(avail, m)
    }
  }
  prob_cols <- paste0("prob_", avail)
  tbl$prob_JM <- rowMeans(tbl[, prob_cols, drop = FALSE])   # 并集动态风险
  tbl$y <- as.integer(tbl$event == 1 & tbl$event_time_d <= L + h)  # horizon 内实际事件
  write.csv(tbl, file.path(DATA, sprintf("jm_pred_L%d_h%d.csv", L, h)), row.names = FALSE)
  ev <- sum(tbl$y); n <- nrow(tbl)
  log(sprintf("输出 jm_pred_L%d_h%d.csv: n=%d 事件=%d (%.1f%%) 并集风险均值 %.3f",
              L, h, n, ev, 100*ev/n, mean(tbl$prob_JM, na.rm=TRUE)))
}

# ---------------------------------------------------------------- 轨迹表型聚类 (随机效应)
log("\n===== 轨迹表型聚类 (基于个体随机效应 b_hat) =====")
if (length(ranefs) >= 1) {
  # 合并各 marker 随机效应(按 id 取交集)，得到多维轨迹表征
  common_ids <- Reduce(intersect, lapply(ranefs, rownames))
  b_all <- do.call(cbind, lapply(ranefs, function(x) x[common_ids, , drop = FALSE]))
  colnames(b_all) <- paste0(rep(names(ranefs), each = ncol(ranefs[[1]])), "_re", seq_len(ncol(ranefs[[1]])))
  b_sc <- scale(b_all)
  log("合并随机效应矩阵:", dim(b_all), " (共同 id:", length(common_ids), ")")

  png(file.path(DATA, "jm_elbow.png"), width = 1200, height = 800, res = 120)
  wss <- sapply(1:8, function(k) kmeans(b_sc, centers = k, nstart = 25, iter.max = 50)$tot.withinss)
  plot(1:8, wss, type = "b", xlab = "k", ylab = "WSS", main = "Elbow: 轨迹表型数")
  dev.off()

  K <- 3
  set.seed(2024)
  cl <- kmeans(b_sc, centers = K, nstart = 50, iter.max = 100)
  clus <- data.frame(id = common_ids, cluster = cl$cluster)
  clus <- merge(clus, ind[, c("id","event_time_d","event")], by = "id")
  write.csv(clus, file.path(DATA, "jm_clusters.csv"), row.names = FALSE)
  log("输出 jm_clusters.csv:", nrow(clus), "例, K =", K)
  log("各簇事件数 (event):")
  print(with(clus, table(cluster, event)))
  log(capture.output(with(clus, table(cluster, event))))

  # 各簇胆红素轨迹均值(可视化)
  sub <- merge(wide[, c("id","tday","Bilirubin")], clus[, "id", drop=FALSE], by="id")
  sub <- merge(sub, clus[, c("id","cluster")], by="id")
  png(file.path(DATA, "jm_cluster_traj.png"), width = 1400, height = 900, res = 120)
  cols <- c("steelblue","darkorange","seagreen","purple","brown")
  plot(NULL, xlim=c(0,14), ylim=c(min(wide$Bilirubin,na.rm=TRUE), max(wide$Bilirubin,na.rm=TRUE)),
       xlab="tday (ICU 入科后天数, log 尺度)", ylab="Bilirubin (log)", main="各轨迹表型胆红素均值轨迹")
  for (k in 1:K) {
    s <- sub[sub$cluster==k, ]
    mtraj <- aggregate(Bilirubin ~ tday, s, mean, na.rm=TRUE)
    lines(mtraj$tday, mtraj$Bilirubin, col=cols[k], lwd=2.5, type="b", pch=16)
  }
  legend("topright", legend=paste0("Cluster ",1:K), col=cols[1:K], lwd=2.5)
  dev.off()
  log("输出 jm_cluster_traj.png")
}

log("\nDONE")
