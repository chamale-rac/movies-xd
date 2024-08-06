# Load necessary libraries
library(tidyverse)
library(lubridate)
library(corrplot)
library(ggpubr)

# Read the dataset
data <- read_csv("./archive/movie_statistic_dataset.csv")

# Save the initial dimensions of the dataset
initial_dim <- dim(data)
initial_dim

# Checking what the data looks like
names(data)

# Renaming cols for easier access
data <- data %>%
    rename_all(tolower) %>%
    rename_all(~ str_replace_all(., "\\$", "")) %>%
    rename_all(str_trim) %>%
    rename_all(~ str_replace_all(., " ", "_"))

# This is so much easier to work with
names(data)
# Checking how the data looks like
head(data)
summary(data)

# Checking for missing values
sum(is.na(data))

# Correlation matrix plot for numeric columns
data_num <- data %>% select_if(is.numeric)

# Correlation matrix values
cor(data_num)
cor(data_num) %>% corrplot::corrplot()

# Birth and death year sanity check
data$director_deathyear <-
    ifelse(data$director_deathyear %in% c("alive", "-"),
        NA, data$director_deathyear
    )
data$director_deathyear <-
    as.numeric(data$director_deathyear)
data$director_birthyear <-
    ifelse(data$director_birthyear %in% c("alive", "-", ""),
        NA, data$director_birthyear
    )
data$director_birthyear <-
    as.numeric(data$director_birthyear)

data$director_age_at_release <-
    as.numeric(format(data$production_date, "%Y")) - data$director_birthyear

# Check NA of columns
colSums(is.na(data))

# Death year have so many missing values, we can drop it
data <- data %>% select(-director_deathyear)


# A new correlation plot
data_num <- data %>% select_if(is.numeric)
cor(data_num) %>% corrplot::corrplot()

# Director age_at_release and
# director_birthyear have no correlation with other variables
# We can drop
data <- data %>% select(-director_age_at_release, -director_birthyear)

# Drop rows with any NA values in categorical columns
data_cat <- data %>%
    select_if(~ !is.numeric(.)) %>%
    drop_na()

chi_square_test <- function(var1, var2) {
    tbl <- table(var1, var2)
    test <- chisq.test(tbl)
    return(test$p.value)
}


cat_vars <- names(data_cat)

chi_square_test <- function(var1, var2) {
    tbl <- table(var1, var2)
    test <- chisq.test(tbl)
    return(test$p.value)
}

cat_vars <- names(data_cat)
p_value_matrix <- matrix(NA,
    ncol = length(cat_vars), nrow = length(cat_vars),
    dimnames = list(cat_vars, cat_vars)
)

for (i in seq_along(cat_vars)) {
    for (j in seq_along(cat_vars)) {
        if (i != j) {
            p_value_matrix[i, j] <-
                chi_square_test(data_cat[[i]], data_cat[[j]])
        }
    }
}

# Chi-Square Reference: https://medium.com/@ritesh.110587/correlation-between-categorical-variables-63f6bd9bf2f7

# Replace NA values with a high p-value (e.g., 1)
p_value_matrix[is.na(p_value_matrix)] <- 1

# Check the matrix numerically
p_value_matrix

cor(p_value_matrix) %>% corrplot::corrplot()

# List of numerical columns
num_vars <- data %>%
    select_if(is.numeric) %>%
    names()

# Function to perform ANOVA and get p-values
anova_test <- function(num_var, cat_var) {
    formula <- as.formula(paste(num_var, "~", cat_var))
    aov_res <- aov(formula, data = data)
    summary(aov_res)[[1]]["Pr(>F)"][1, 1]
}

# Perform ANOVA for each numerical variable against 'genres'
p_values_anova <- sapply(num_vars, function(x) anova_test(x, "genres"))

# Display p-values
p_values_anova

# Anova Reference: http://www.sthda.com/english/wiki/one-way-anova-test-in-r

# director_name frequency chart
data %>%
    count(director_name) %>%
    ggplot(aes(x = reorder(director_name, n), y = n)) +
    geom_bar(stat = "identity") +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
# How many unique directors do we have?
length(unique(data$director_name))

# Selected features
selected_feature <- c(
    "approval_index",
    "movie_averagerating",
    "production_budget",
    "runtime_minutes",
    "worldwide_gross"
)

# Boxplot for selected features
data %>%
    select(selected_feature) %>%
    gather(key = "feature", value = "value") %>%
    ggplot(aes(x = feature, y = value)) +
    geom_boxplot() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Boxplot for selected features
data %>%
    select(selected_feature) %>%
    gather(key = "feature", value = "value") %>%
    ggplot(aes(x = feature, y = value)) +
    geom_boxplot() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Normalize numerical features
data[selected_feature] <- scale(data[selected_feature])

# Genres one hot encoding
genres_split <- str_split(data$genres, ",")
unique_genres <- unique(unlist(genres_split))

for (genre in unique_genres) {
    data[[genre]] <-
        sapply(genres_split, function(genres) ifelse(genre %in% genres, 1, 0))
}

# Generate the final dataset using unique genres and selected features
final_data <- data %>%
    select(selected_feature, unique_genres)

# All genres to lower case
final_data <- final_data %>%
    rename_all(tolower)

names(final_data)

# Save the final dataset
write_csv(final_data, "./archive/final_data.csv")

# Percentage of data retained
final_dim <- dim(final_data)

cat("Percentage of data retained: ", final_dim[1] / initial_dim[1] * 100, "%\n")

# Genres frequency chart
final_data %>%
    gather(key = "genre", value = "value", -selected_feature) %>%
    filter(value == 1) %>%
    count(genre) %>%
    ggplot(aes(x = reorder(genre, n), y = n)) +
    geom_bar(stat = "identity") +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))

