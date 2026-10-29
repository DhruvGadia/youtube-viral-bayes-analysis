# What makes a YouTube video go viral? A Bayesian mini-project
# Data: Kaggle Trending YouTube Video Statistics, India subset
# https://www.kaggle.com/datasets/datasnaek/youtube-new
#
# Run this script after setting raw_dir to the folder containing INvideos.csv
# and IN_category_id.json. It needs jsonlite only for category names:
# install.packages("jsonlite")

set.seed(2026)
raw_dir <- "C:/Users/HP/Documents/dataset"  # change if necessary
out_dir <- "outputs"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# 1. Import and clean the daily trending snapshots
raw <- read.csv(file.path(raw_dir, "INvideos.csv"),
                stringsAsFactors = FALSE, fileEncoding = "UTF-8",
                check.names = FALSE)
categories <- jsonlite::fromJSON(file.path(raw_dir, "IN_category_id.json"))$items
category_lookup <- setNames(categories$snippet$title, categories$id)

raw$country <- "IN"
raw$trending_date_parsed <- as.Date(raw$trending_date, format = "%y.%d.%m")
raw$publish_datetime <- as.POSIXct(raw$publish_time, format = "%Y-%m-%dT%H:%M:%OSZ", tz = "UTC")
raw$category <- unname(category_lookup[as.character(raw$category_id)])
raw$category[is.na(raw$category)] <- "Unknown"

numeric_fields <- c("views", "likes", "dislikes", "comment_count")
for (field in numeric_fields) raw[[field]] <- suppressWarnings(as.numeric(raw[[field]]))
raw <- raw[!is.na(raw$video_id) & raw$views >= 0 & raw$likes >= 0 &
           raw$dislikes >= 0 & raw$comment_count >= 0 & !is.na(raw$trending_date_parsed), ]

raw$title_length <- nchar(raw$title)
raw$tag_count <- ifelse(is.na(raw$tags) | raw$tags == "[none]" | raw$tags == "", 0,
                         lengths(strsplit(raw$tags, "\\|", fixed = FALSE)))
raw$comments_disabled <- toupper(raw$comments_disabled) == "TRUE"
raw$ratings_disabled <- toupper(raw$ratings_disabled) == "TRUE"
raw$likes_per_view <- raw$likes / pmax(raw$views, 1)
raw$comments_per_view <- raw$comment_count / pmax(raw$views, 1)
raw$age_hours <- as.numeric(difftime(as.POSIXct(raw$trending_date_parsed, tz = "UTC"),
                                     raw$publish_datetime, units = "hours"))

# Save every cleaned daily snapshot, then retain one first-trending row per video.
raw <- raw[order(raw$video_id, raw$trending_date_parsed), ]
write.csv(raw, file.path(out_dir, "INvideos_cleaned_daily_snapshots.csv"), row.names = FALSE)
first <- raw[!duplicated(raw$video_id), ]

# 2. Train/test split and target definition
# Viral = top quartile of views at the first observed trending appearance.
initial_class <- ifelse(first$views >= quantile(first$views, .75, na.rm = TRUE), "Viral", "Not viral")
train_index <- unlist(tapply(seq_len(nrow(first)), initial_class,
                             function(i) sample(i, floor(0.70 * length(i)))))
first$split <- ifelse(seq_len(nrow(first)) %in% train_index, "Train", "Test")
viral_cutoff <- quantile(first$views[first$split == "Train"], .75, na.rm = TRUE)
first$viral <- ifelse(first$views >= viral_cutoff, "Viral", "Not viral")

# Feature bins use training medians only, preventing information leakage.
train <- first[first$split == "Train", ]
med_like <- median(train$likes_per_view, na.rm = TRUE)
med_comment <- median(train$comments_per_view, na.rm = TRUE)
med_title <- median(train$title_length, na.rm = TRUE)
first$like_rate <- ifelse(first$likes_per_view >= med_like, "High", "Low")
first$comment_rate <- ifelse(first$comments_per_view >= med_comment, "High", "Low")
first$title_length_band <- ifelse(first$title_length >= med_title, "Long", "Short")
first$publish_hour <- as.integer(format(first$publish_datetime, "%H", tz = "UTC"))
first$publish_period <- cut(first$publish_hour, breaks = c(-1, 5, 11, 17, 23),
                            labels = c("Night", "Morning", "Afternoon", "Evening"))

# 3. Naive Bayes with Laplace smoothing: P(Viral|x) proportional to P(Viral) product P(x_j|Viral)
features <- c("category", "like_rate", "comment_rate", "title_length_band", "publish_period")
train <- first[first$split == "Train", ]
test <- first[first$split == "Test", ]
prior_viral <- mean(train$viral == "Viral")

conditional_probability <- function(feature, value, target) {
  values <- unique(first[[feature]])
  values <- values[!is.na(values)]
  target_rows <- train$viral == target
  count <- sum(train[[feature]][target_rows] == value, na.rm = TRUE)
  (count + 1) / (sum(target_rows) + length(values))
}

posterior_viral <- function(row) {
  p_v <- prior_viral
  p_nv <- 1 - prior_viral
  for (feature in features) {
    value <- row[[feature]]
    p_v <- p_v * conditional_probability(feature, value, "Viral")
    p_nv <- p_nv * conditional_probability(feature, value, "Not viral")
  }
  p_v / (p_v + p_nv)
}

test$posterior_viral <- apply(test[, features, drop = FALSE], 1,
                              function(x) posterior_viral(as.list(x)))
test$prediction <- ifelse(test$posterior_viral >= 0.50, "Viral", "Not viral")
accuracy <- mean(test$prediction == test$viral)
precision <- sum(test$prediction == "Viral" & test$viral == "Viral") /
             max(1, sum(test$prediction == "Viral"))
recall <- sum(test$prediction == "Viral" & test$viral == "Viral") /
          max(1, sum(test$viral == "Viral"))

# 4. Transparent Bayes-theorem evidence table
evidence <- do.call(rbind, lapply(features, function(feature) {
  values <- sort(unique(first[[feature]]))
  values <- values[!is.na(values)]
  do.call(rbind, lapply(values, function(value) {
    feature_rows <- train[[feature]] == value
    data.frame(feature = feature, value = value,
               videos_in_train = sum(feature_rows),
               p_feature_given_viral = conditional_probability(feature, value, "Viral"),
               p_feature_given_not_viral = conditional_probability(feature, value, "Not viral"),
               posterior_p_viral_given_feature = mean(train$viral[feature_rows] == "Viral"),
               stringsAsFactors = FALSE)
  }))
}))

metrics <- data.frame(metric = c("Unique videos", "Daily snapshots", "Training videos",
                                  "Test videos", "Viral views cutoff", "Viral prior",
                                  "Test accuracy", "Test precision", "Test recall"),
                      value = c(nrow(first), nrow(raw), nrow(train), nrow(test), viral_cutoff,
                                prior_viral, accuracy, precision, recall))

write.csv(first, file.path(out_dir, "youtube_in_first_trending_videos.csv"), row.names = FALSE)
write.csv(test, file.path(out_dir, "youtube_in_bayes_test_predictions.csv"), row.names = FALSE)
write.csv(evidence, file.path(out_dir, "youtube_in_bayes_evidence.csv"), row.names = FALSE)
write.csv(metrics, file.path(out_dir, "youtube_in_bayes_metrics.csv"), row.names = FALSE)

# Console output for the project presentation
print(metrics)
print(evidence[order(-evidence$posterior_p_viral_given_feature), ][1:10, ])

