##############################################################
# PFFR (Penalized Function-on-Function Regression) using refund
# Model: Y(t) = ff(PM25, xind=s) + ff(WIND_U) + ff(WIND_V) +
#               ff(RH) + ff(TEMP) + s(lat, lon)
# Evaluation: NCU RMSE (hourly RMSE averaged over 72 hours)
##############################################################

cat("Loading libraries...\n")
suppressPackageStartupMessages({
  library(refund)
  library(mgcv)
})

base_path <- "C:/Users/user/Desktop/HW-NCHU/meeting/ccproject_deeponet/data"

# ============================================================
# DATA LOADING
# ============================================================
cat("Loading data...\n")

# Load FPCA processed features
load_var <- function(varname, filename) {
  df <- read.csv(file.path(base_path, "fpca_processed", filename))
  df$PublishTime <- as.POSIXct(df$PublishTime)
  # Get station columns (all except metadata)
  meta_cols <- c("date", "Time", "year", "SubjectID", "PublishTime")
  station_cols <- setdiff(names(df), meta_cols)
  # Melt to long format
  long <- reshape(df[, c("PublishTime", station_cols)],
                  direction = "long",
                  varying = station_cols,
                  v.names = varname,
                  timevar = "Station",
                  times = station_cols)
  long$id <- NULL
  rownames(long) <- NULL
  return(long)
}

pm25 <- load_var("PM25", "PM2.5_FPCA_2025.csv")
windu <- load_var("WIND_U", "WIND_U_FPCA_2025.csv")
windv <- load_var("WIND_V", "WIND_V_FPCA_2025.csv")
rh <- load_var("RH", "RH_FPCA_2025.csv")
temp <- load_var("TEMP", "AMB_TEMP_FPCA_2025.csv")

# Merge
df_all <- merge(pm25, windu, by = c("PublishTime", "Station"))
df_all <- merge(df_all, windv, by = c("PublishTime", "Station"))
df_all <- merge(df_all, rh, by = c("PublishTime", "Station"))
df_all <- merge(df_all, temp, by = c("PublishTime", "Station"))

# Load raw PM2.5 for test evaluation
raw <- read.csv(file.path(base_path, "raw", "PM2.5.csv"))
raw$PublishTime <- as.POSIXct(raw$PublishTime)
meta_cols_raw <- c("PublishTime")
station_cols_raw <- setdiff(names(raw), meta_cols_raw)
raw_long <- reshape(raw[, c("PublishTime", station_cols_raw)],
                    direction = "long",
                    varying = station_cols_raw,
                    v.names = "PM25_Raw",
                    timevar = "Station",
                    times = station_cols_raw)
raw_long$id <- NULL
rownames(raw_long) <- NULL

df_all <- merge(df_all, raw_long, by = c("PublishTime", "Station"), all.x = TRUE)
df_all <- df_all[order(df_all$Station, df_all$PublishTime), ]

# Station info
station_info <- read.csv(file.path(base_path, "station_info", "station .csv"),
                         fileEncoding = "UTF-8-BOM")
coords <- setNames(
  data.frame(station_info$lat, station_info$lon),
  c("lat", "lon")
)
rownames(coords) <- station_info$SITE_NAME

cat("Data loaded. Building windows...\n")

# ============================================================
# BUILD SLIDING WINDOWS
# ============================================================
INPUT_HOURS <- 24
OUTPUT_HOURS <- 72
TOTAL_WINDOW <- INPUT_HOURS + OUTPUT_HOURS
STEP_SIZE <- 24
SPLIT_DATE <- as.POSIXct("2025-01-01")
TRAIN_START <- as.POSIXct("2018-01-01")
TEST_END <- as.POSIXct("2025-11-30")

stations <- unique(df_all$Station)
# Filter to stations with coordinates
stations <- stations[stations %in% rownames(coords)]

train_list <- list()
test_list <- list()

for (sn in stations) {
  df_s <- df_all[df_all$Station == sn, ]
  df_s <- df_s[order(df_s$PublishTime), ]
  n <- nrow(df_s)
  ns <- n - TOTAL_WINDOW + 1
  if (ns <= 0) next

  lat <- coords[sn, "lat"]
  lon <- coords[sn, "lon"]

  for (i in seq(1, ns, by = STEP_SIZE)) {
    ct <- df_s$PublishTime[i]
    # Input: rows i to i+23 (24 hours), 5 variables
    x_pm25 <- df_s$PM25[i:(i + INPUT_HOURS - 1)]
    x_windu <- df_s$WIND_U[i:(i + INPUT_HOURS - 1)]
    x_windv <- df_s$WIND_V[i:(i + INPUT_HOURS - 1)]
    x_rh <- df_s$RH[i:(i + INPUT_HOURS - 1)]
    x_temp <- df_s$TEMP[i:(i + INPUT_HOURS - 1)]

    if (any(is.na(c(x_pm25, x_windu, x_windv, x_rh, x_temp)))) next

    if (ct >= TRAIN_START & ct < SPLIT_DATE) {
      # Output: FPCA smoothed for training
      y <- df_s$PM25[(i + INPUT_HOURS):(i + TOTAL_WINDOW - 1)]
      if (any(is.na(y))) next
      train_list[[length(train_list) + 1]] <- list(
        pm25 = x_pm25, windu = x_windu, windv = x_windv,
        rh = x_rh, temp = x_temp, y = y, lat = lat, lon = lon
      )
    } else if (ct >= SPLIT_DATE & ct <= TEST_END - TOTAL_WINDOW * 3600) {
      # Output: raw for testing
      y_raw <- df_s$PM25_Raw[(i + INPUT_HOURS):(i + TOTAL_WINDOW - 1)]
      if (all(is.na(y_raw))) next
      test_list[[length(test_list) + 1]] <- list(
        pm25 = x_pm25, windu = x_windu, windv = x_windv,
        rh = x_rh, temp = x_temp, y_raw = y_raw, lat = lat, lon = lon
      )
    }
  }
}

N_train <- length(train_list)
N_test <- length(test_list)
cat(sprintf("Train: %d | Test: %d\n", N_train, N_test))

# Convert to matrices for pffr
# Input: each variable is a (N, 24) matrix
# Output: Y is a (N, 72) matrix
s_grid <- 1:24
t_grid <- 1:72

make_matrix <- function(lst, field) {
  do.call(rbind, lapply(lst, function(x) x[[field]]))
}

X_pm25_train <- make_matrix(train_list, "pm25")   # (N_train, 24)
X_windu_train <- make_matrix(train_list, "windu")
X_windv_train <- make_matrix(train_list, "windv")
X_rh_train <- make_matrix(train_list, "rh")
X_temp_train <- make_matrix(train_list, "temp")
Y_train <- make_matrix(train_list, "y")            # (N_train, 72)
lat_train <- sapply(train_list, function(x) x$lat)
lon_train <- sapply(train_list, function(x) x$lon)

X_pm25_test <- make_matrix(test_list, "pm25")
X_windu_test <- make_matrix(test_list, "windu")
X_windv_test <- make_matrix(test_list, "windv")
X_rh_test <- make_matrix(test_list, "rh")
X_temp_test <- make_matrix(test_list, "temp")
Y_test <- make_matrix(test_list, "y_raw")
lat_test <- sapply(test_list, function(x) x$lat)
lon_test <- sapply(test_list, function(x) x$lon)

cat(sprintf("Y_train: %d x %d\n", nrow(Y_train), ncol(Y_train)))
cat(sprintf("Y_test:  %d x %d\n", nrow(Y_test), ncol(Y_test)))

# ============================================================
# SUBSAMPLE FOR PFFR (full data too large for mgcv)
# ============================================================
# pffr with ff() on 176K samples is infeasible - subsample
set.seed(42)
N_sub <- 5000
cat(sprintf("Subsampling to %d for pffr fitting...\n", N_sub))
idx_sub <- sample(N_train, N_sub)

# Build data list for pffr
train_data <- list(
  Y = Y_train[idx_sub, ],
  X_pm25 = X_pm25_train[idx_sub, ],
  X_windu = X_windu_train[idx_sub, ],
  X_windv = X_windv_train[idx_sub, ],
  X_rh = X_rh_train[idx_sub, ],
  X_temp = X_temp_train[idx_sub, ],
  lat = lat_train[idx_sub],
  lon = lon_train[idx_sub]
)

# ============================================================
# FIT PFFR with ff() terms
# ============================================================
cat("Fitting pffr model...\n")
cat("  This may take a while...\n")
t0 <- proc.time()

# ff() = function-on-function smooth term
# s(lat, lon) = spatial smooth
# Use fewer basis to keep computation feasible
fit <- pffr(
  Y ~ ff(X_pm25, xind = s_grid, basistype = "s", splinepars = list(bs = "ps", k = 8)) +
       ff(X_windu, xind = s_grid, basistype = "s", splinepars = list(bs = "ps", k = 5)) +
       ff(X_windv, xind = s_grid, basistype = "s", splinepars = list(bs = "ps", k = 5)) +
       ff(X_rh, xind = s_grid, basistype = "s", splinepars = list(bs = "ps", k = 5)) +
       ff(X_temp, xind = s_grid, basistype = "s", splinepars = list(bs = "ps", k = 5)) +
       s(lat, lon, k = 15),
  yind = t_grid,
  data = train_data,
  bs.yindex = list(bs = "ps", k = 10)
)

elapsed <- (proc.time() - t0)[3]
cat(sprintf("  pffr fitted in %.1f seconds\n", elapsed))

# ============================================================
# PREDICT ON FULL TEST SET
# ============================================================
cat("Predicting on test set...\n")
test_data <- list(
  X_pm25 = X_pm25_test,
  X_windu = X_windu_test,
  X_windv = X_windv_test,
  X_rh = X_rh_test,
  X_temp = X_temp_test,
  lat = lat_test,
  lon = lon_test
)

Y_pred <- predict(fit, newdata = test_data)

# ============================================================
# NCU RMSE EVALUATION
# ============================================================
cat("\nEvaluating NCU RMSE...\n")
hourly_rmse <- numeric(72)
hourly_mae <- numeric(72)
for (h in 1:72) {
  valid <- !is.na(Y_test[, h])
  if (sum(valid) > 0) {
    err <- Y_test[valid, h] - Y_pred[valid, h]
    hourly_rmse[h] <- sqrt(mean(err^2))
    hourly_mae[h] <- mean(abs(err))
  } else {
    hourly_rmse[h] <- NA
    hourly_mae[h] <- NA
  }
}

ncu_rmse <- mean(hourly_rmse, na.rm = TRUE)
ncu_mae <- mean(hourly_mae, na.rm = TRUE)
d1 <- mean(hourly_rmse[1:24], na.rm = TRUE)
d2 <- mean(hourly_rmse[25:48], na.rm = TRUE)
d3 <- mean(hourly_rmse[49:72], na.rm = TRUE)

cat(sprintf("\n========================================\n"))
cat(sprintf("PFFR Results (refund::pffr with ff())\n"))
cat(sprintf("========================================\n"))
cat(sprintf("  Subsample size: %d\n", N_sub))
cat(sprintf("  NCU RMSE = %.4f\n", ncu_rmse))
cat(sprintf("  NCU MAE  = %.4f\n", ncu_mae))
cat(sprintf("  Day 1 (1-24hr):  %.4f\n", d1))
cat(sprintf("  Day 2 (25-48hr): %.4f\n", d2))
cat(sprintf("  Day 3 (49-72hr): %.4f\n", d3))
cat(sprintf("  Fit time: %.1f seconds\n", elapsed))
cat("Done!\n")
