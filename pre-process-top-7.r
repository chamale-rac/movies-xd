# Load necessary libraries
library(tidyverse)
library(lubridate)
library(corrplot)
library(ggpubr)

# Read the dataset
data <- read_csv("./archive/movie_statistic_dataset.csv")

# Save the initial dimensions of the dataset
initial_dim <- dim(data)

# Renaming cols for easier access
data <- data %>%
    rename_all(tolower) %>%
    rename_all(~ str_replace_all(., "\\$", "")) %>%
    rename_all(str_trim) %>%
    rename_all(~ str_replace_all(., " ", "_"))

# Birth and death year sanity check
data$director_deathyear <- ifelse(data$director_deathyear %in% c("alive", "-"), NA, data$director_deathyear)
data$director_deathyear <- as.numeric(data$director_deathyear)
data$director_birthyear <- ifelse(data$director_birthyear %in% c("alive", "-", ""), NA, data$director_birthyear)
data$director_birthyear <- as.numeric(data$director_birthyear)

data$director_age_at_release <- as.numeric(format(data$production_date, "%Y")) - data$director_birthyear

# Drop director_deathyear, director_age_at_release, and director_birthyear
data <- data %>% select(-director_deathyear, -director_age_at_release, -director_birthyear)

# Drop rows with any NA values in categorical columns
data <- data %>% drop_na()

# Selected features
selected_feature <- c(
    "approval_index",
    "movie_averagerating",
    "production_budget",
    "runtime_minutes",
    "worldwide_gross"
)

# Normalize numerical features
data[selected_feature] <- scale(data[selected_feature])

# Genres one hot encoding
genres_split <- str_split(data$genres, ",")
unique_genres <- unique(unlist(genres_split))

for (genre in unique_genres) {
    data[[genre]] <- sapply(genres_split, function(genres) ifelse(genre %in% genres, 1, 0))
}

# Generate the final dataset using unique genres and selected features
final_data <- data %>%
    select(selected_feature, unique_genres)

# Calculate genre frequencies
genre_freq <- final_data %>%
    select(unique_genres) %>%
    summarise_all(sum) %>%
    gather(key = "genre", value = "frequency") %>%
    arrange(desc(frequency))

# Get top 7 genres
top_genres <- genre_freq$genre[1:7]

# Create a dataset with only top 7 genres
final_data_top_genres <- final_data %>%
    select(selected_feature, top_genres)

# Ensure equal frequency for top genres
min_freq <- genre_freq$frequency[1:7] %>% min()

equal_freq_data <- final_data_top_genres %>%
    gather(key = "genre", value = "value", -selected_feature) %>%
    filter(value == 1) %>%
    group_by(genre) %>%
    slice(1:min_freq) %>%
    ungroup() %>%
    spread(key = "genre", value = "value", fill = 0)

# All variables names to lower case
equal_freq_data <- equal_freq_data %>%
    rename_all(tolower)

# Save the final dataset with top 7 genres having equal frequency
write_csv(equal_freq_data, "./archive/final_data_top_7_genres.csv")

# Percentage of data retained
final_dim <- dim(equal_freq_data)
cat("Percentage of data retained: ", final_dim[1] / initial_dim[1] * 100, "%\n")

# Genres frequency chart for top 7 genres
equal_freq_data %>%
    gather(key = "genre", value = "value", -selected_feature) %>%
    filter(value == 1) %>%
    count(genre) %>%
    ggplot(aes(x = reorder(genre, n), y = n)) +
    geom_bar(stat = "identity") +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
