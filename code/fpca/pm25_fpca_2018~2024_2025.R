#install.packages("fdapace")
library(tidyr)
library(dplyr)
library(lubridate)
library(fdapace)

data <- read.csv("C:\\Users\\user\\Desktop\\HW-NCHU\\meeting\\GPR\\data\\RH.csv",fileEncoding = "UTF-8")
all_stations <- colnames(data)
all_stations <- all_stations[all_stations != "PublishTime"]
#data
all_smoothed_results <- list()

for (station in all_stations){
    cat("處理站點:", station, "\n")
    
    tryCatch({
    
        cleaned_data <- data %>%
            mutate(PublishTime = parse_date_time(PublishTime, orders = "ymd HMS")) %>%
            filter(year(PublishTime) >= 2018 & year(PublishTime) <= 2025) %>%
            mutate(date = as.Date(PublishTime)) %>%
            mutate(Time = hour(PublishTime)) %>%
            mutate(SubjectID = as.numeric(factor(date))) %>%
            dplyr::select(PublishTime, SubjectID, Time, date, all_of(station)) %>%
            arrange(PublishTime) %>%
            drop_na()
        
        # 分割點：設定為 2021-10-01
        split_date <- as.Date("2025-01-01")

        # 分割訓練/測試集 (2024年為分界點)
        train_data <- cleaned_data %>%
            filter(date < split_date)
        
        test_data <- cleaned_data %>%
            filter(date >= split_date)

        # 準備FPCA輸入
        train_Ly <- train_data %>%
            group_by(SubjectID) %>%
            summarise(y = list(.data[[station]])) %>%
            pull(y)
        
        train_Lt <- train_data %>%
            group_by(SubjectID) %>%
            summarise(t = list(Time)) %>%
            pull(t)
        
        test_Ly <- test_data %>%
            group_by(SubjectID) %>%
            summarise(y = list(.data[[station]])) %>%
            pull(y)
        
        test_Lt <- test_data %>%
            group_by(SubjectID) %>%
            summarise(t = list(Time)) %>%
            pull(t)
        
        # FPCA
        fpca_train <- FPCA(
            Ly = train_Ly,
            Lt = train_Lt,
            optns = list(dataType = 'Sparse', FVEthreshold = 0.999, nRegGrid = 24,kernel = "epan")
        )

        pm25_train_smooth <- fitted(fpca_train)
        
        pred_result <- predict(fpca_train,newLy = test_Ly,newLt = test_Lt)
        scores <- pred_result$scores
        mu <- fpca_train$mu
        phi <- fpca_train$phi
        pm25_test_smooth <- matrix(mu, nrow = nrow(scores), ncol = length(mu), byrow = TRUE) +
                                scores %*% t(phi[, 1:ncol(scores)])
                                
        # 合併 train + test 的平滑結果
        all_smooth <- rbind(pm25_train_smooth, pm25_test_smooth)
        all_subject_ids <- c(unique(train_data$SubjectID), unique(test_data$SubjectID))
        
        # 儲存結果
        all_smoothed_results[[station]] <- data.frame(
            SubjectID = rep(all_subject_ids, each = 24),
            Time = rep(0:23, times = length(all_subject_ids)),
            smooth_value = as.vector(t(all_smooth))
        )
        
        cat(station, "- 完成\n")
    
    }, error = function(e) {
        cat("====================失敗\n") # 失敗印出失敗，並自動跳下一個
    })
}


# 建立基礎的 PublishTime 對照表
result_df <- data %>%
    mutate(PublishTime = parse_date_time(PublishTime, orders = "ymd HMS")) %>%
    filter(year(PublishTime) >= 2018 & year(PublishTime) <= 2025) %>%
    mutate(date = as.Date(PublishTime)) %>%
    mutate(Time = hour(PublishTime)) %>%
    mutate(SubjectID = as.numeric(factor(date))) %>%
    dplyr::select(PublishTime, SubjectID, Time) %>%
    arrange(PublishTime)

# 為每個站點加入平滑後的值
for (station in all_stations) {
    if (!is.null(all_smoothed_results[[station]])) {
        
        # 情況 A: 有東西 (非 NULL)，才執行 join
        result_df <- result_df %>%
            left_join(all_smoothed_results[[station]], by = c("SubjectID", "Time")) %>%
            rename(!!station := smooth_value)
            
    } else {
        
        # 情況 B: 是 NULL (代表之前計算失敗了)，直接填入 NA
        # 如果不寫這行 else，left_join 就會報錯
        result_df[[station]] <- NA 
    }
}

# 只保留 PublishTime 和各站點
result_df <- result_df %>%
    mutate(PublishTime = format(PublishTime, "%Y-%m-%d %H:%M:%S")) %>%
    dplyr::select(PublishTime, any_of(all_stations))

# 輸出
write.csv(result_df, "C:\\Users\\user\\Desktop\\HW-NCHU\\meeting\\GPR\\data\\RH_FPCA_2025.csv", 
          row.names = FALSE, fileEncoding = "UTF-8")

cat("\n所有站點處理完成!\n")
cat("結果已儲存至: PM2.5_FPCA.csv\n")
cat("總共", length(all_stations), "個站點\n")
cat("資料列數:", nrow(result_df), "\n")
cat("格式: PublishTime + 各站點平滑值\n")



library(dplyr)
file_path <- "C:\\Users\\user\\Desktop\\HW-NCHU\\meeting\\GPR\\data\\WIND_V_FPCA.csv"
data <- read.csv(file_path, fileEncoding = "UTF-8", stringsAsFactors = FALSE)
target_cols <- setdiff(names(data), c("PublishTime", "SubjectID", "date", "Time"))
suppressWarnings({
    data[target_cols] <- lapply(data[target_cols], as.numeric)
})
na_counts <- colSums(is.na(data[target_cols]))
has_na <- na_counts[na_counts >= 0]

if(length(has_na) >= 0) {
    cat("以下欄位含有 NA (包含空格或非數字)，及其數量：\n")
    print(has_na)
} else {
    cat("恭喜！所有指定欄位皆為有效數字，沒有 NA。\n")
}


