# Redo only the JM-mv trajectory clustering (taking the posterior means b of the subject-specific
# random effects directly from jm_mv_fit.rds)
suppressPackageStartupMessages({library(nlme); library(JMbayes2)})
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
log <- function(...) { cat(..., "\n") }
jm_mv <- readRDS(file.path(DATA, "jm_mv_fit.rds"))
wide <- read_csv_safe(file.path(DATA, "jm_wide.csv"))
surv_id <- read_csv_safe(file.path(DATA, "jm_surv.csv"))
ind <- surv_id[!duplicated(surv_id$id), ]

b_all <- jm_mv$statistics$Mean$b
ids_b <- as.character(levels(jm_mv$model_data$idT))
stopifnot(nrow(b_all) == length(ids_b))
rownames(b_all) <- ids_b
b_sc <- scale(b_all)
log("Random-effects matrix:", paste(dim(b_all), collapse="x"))

png(file.path(DATA, "jm_mv_elbow.png"), width = 1200, height = 800, res = 120)
wss <- sapply(1:8, function(k) kmeans(b_sc, centers = k, nstart = 25, iter.max = 50)$tot.withinss)
plot(1:8, wss, type = "b", xlab = "k", ylab = "WSS",
     main = "Elbow: number of JM-mv trajectory phenotypes")
dev.off()

K <- 3
cl <- kmeans(b_sc, centers = K, nstart = 50, iter.max = 100)
clus <- data.frame(id = as.integer(ids_b), cluster = cl$cluster)
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
log("JM_MV CLUSTER DONE")
