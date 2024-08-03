# Load necessary libraries
library(tidyverse)
library(lubridate)
library(caret)

# Read the dataset
data <- read_csv("./archive/movie_statistic_dataset.csv")

# Save the initial dimensions of the dataset
initial_dim <- dim(data)

# Display the first few rows and summary of the dataset
head(data)
summary(data)

# Check for missing values
sum(is.na(data))

# Data Cleaning
# Convert production_date to Date type
data$production_date <- as.Date(data$production_date, format = "%Y-%m-%d")

# Replace 'alive' and '-' with NA in director_deathYear and convert to numeric
data$director_deathYear <-
    ifelse(data$director_deathYear %in% c("alive", " -"),
        NA, data$director_deathYear
    )
data$director_deathYear <- as.numeric(data$director_deathYear)

# Boxplot for director_deathYear
boxplot(data$director_deathYear, main = "Director Death Year")

# Convert budget and gross columns to numeric
data$production_budget <-
    as.numeric(gsub("[\\$,]", "", data$`Production budget $`))
data$domestic_gross <- as.numeric(gsub("[\\$,]", "", data$`Domestic gross $`))
data$worldwide_gross <- as.numeric(gsub("[\\$,]", "", data$`Worldwide gross $`))

# Clean and convert director_birthYear
data$director_birthYear <-
    ifelse(data$director_birthYear %in% c("alive", " -", ""),
        NA, data$director_birthYear
    )
data$director_birthYear <- as.numeric(data$director_birthYear)

# Calculate director's age at release
data$director_age_at_release <-
    as.numeric(format(data$production_date, "%Y")) - data$director_birthYear

# Boxplot for director_age_at_release
boxplot(data$director_age_at_release, main = "Director Age at Release")

# Check NA values in director_age_at_release and per column
sum(is.na(data$director_age_at_release))
colSums(is.na(data))

# Remove director_deathYear due to many NA values
data <- data %>% select(-director_deathYear)

# Check for outliers in numerical columns and filter accordingly
boxplot(data$runtime_minutes, main = "Runtime Minutes")
boxplot(data$movie_average_rating, main = "Average Rating")
data <- data %>% filter(movie_average_rating >= 4)
dim(data)

boxplot(data$movie_number_of_votes, main = "Number of Votes")
votes_threshold <- quantile(data$movie_number_of_votes, 0.95)
data <- data %>% filter(movie_number_of_votes <= votes_threshold)
dim(data)

boxplot(data$approval_index, main = "Approval Index")

boxplot(data$production_budget, main = "Production Budget")
budget_threshold <- quantile(data$production_budget, 0.95)
data <- data %>% filter(production_budget <= budget_threshold)
dim(data)

boxplot(data$domestic_gross, main = "Domestic Gross")
domestic_threshold <- quantile(data$domestic_gross, 0.95)
data <- data %>% filter(domestic_gross <= domestic_threshold)
dim(data)

boxplot(data$worldwide_gross, main = "Worldwide Gross")
worldwide_threshold <- quantile(data$worldwide_gross, 0.95)
data <- data %>% filter(worldwide_gross <= worldwide_threshold)
dim(data)

# Remove rows with any missing values
data <- na.omit(data)
dim(data)

# Normalize numerical features
numerical_features <- c(
    "runtime_minutes", "movie_average_rating", "movie_number_of_votes",
    "approval_index", "production_budget", "domestic_gross",
    "worldwide_gross", "director_age_at_release"
)

data[numerical_features] <- scale(data[numerical_features])

# Encode genres for classification (one-hot encoding)
genres_split <- str_split(data$genres, ",")
unique_genres <- unique(unlist(genres_split))

# Plot distribution of genres
genre_counts <-
    sapply(unique_genres, function(genre) sum(sapply(genres_split, function(genres) genre %in% genres)))
barplot(genre_counts,
    names.arg = unique_genres,
    las = 2, main = "Genre Distribution"
)

# Remove uncommon genres representing less than 3% of the dataset
threshold <- 0.05 * nrow(data)
unique_genres <- unique_genres[genre_counts > threshold]

# Display removed genres
removed_genres <- setdiff(unique_genres_previous, unique_genres)
cat("Removed Genres: ", removed_genres, "\n")

for (genre in unique_genres) {
    data[[genre]] <-
        sapply(genres_split, function(genres) ifelse(genre %in% genres, 1, 0))
}

# Select features for the model
selected_features <- c(
    "runtime_minutes", "movie_average_rating", "movie_number_of_votes",
    "approval_index", "production_budget", "domestic_gross",
    "worldwide_gross", "director_age_at_release"
)


# Prepare the final dataset for modeling
final_data <- data %>%
    select(all_of(selected_features), unique_genres)

# All column names to lowercase
colnames(final_data) <- tolower(colnames(final_data))

# Save the cleaned and prepared dataset
write_csv(final_data, "./archive/prepared_movie_statistic_dataset.csv")

# Check dimensions of the final dataset
final_dim <- dim(final_data)

# Calculate and display the reduced dataset percentage
reduction_percentage <- final_dim[1] / initial_dim[1]
cat("Reduction Percentage: ", reduction_percentage * 100, "%\n")
