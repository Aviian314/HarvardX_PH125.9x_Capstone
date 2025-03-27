##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)
################################################################################

# Additional Libraries
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")

library(matrixStats)

################################################################################

# Functions
RMSE <- function(trueValue, predictedValue) {
  sqrt(mean((trueValue - predictedValue)^2, na.rm=TRUE))
}
################################################################################



##### Data Cleaning #####
# Validate the data does not have any null entries.
hasNulls <- any(is.na(edx))
hasNulls



##### Summary Statistics #####
# Print summary statistics for data exploration
summary(edx$rating)
table(edx$rating)

# Plot the distribution of movie ratings.
edx %>%
  ggplot(aes(x = rating)) +
  geom_bar(fill = "steelblue", color = "black") +
  labs(title = "Distribution of Movie Ratings", x = "Rating", y = "Number of Ratings")



##### Movie Popularity Effects #####
# Get each movie's average rating
movie_avg_ratings <- edx %>%
  group_by(movieId) %>%
  summarize(avg_rating = mean(rating), count = n())

# Plot movie popularity (number of ratings) vs. average rating
movie_avg_ratings %>%
  ggplot(aes(x = count, y = avg_rating)) +
  geom_point(alpha = 0.5, color = "steelblue") +
  geom_smooth(method = "lm", color = "firebrick", se = FALSE) +
  scale_x_log10() +
  labs(title = "Number of Ratings vs. Average Rating",
       x = "Number of Ratings", y = "Average Rating")



##### User Rating Bias #####
# Overall movie average rating
avg_rating <- mean(edx$rating)

# Gets each users ratings as an average
user_bias <- edx %>%
  group_by(userId) %>%
  summarize(user_avg_rating = mean(rating), user_bias = user_avg_rating - avg_rating)

# Plot the user's average rating in a histogram
user_bias %>%
  ggplot(aes(x = user_bias)) +
  geom_histogram(binwidth = 0.1, fill = "steelblue", color = "black") +
  labs(title = "User Bias Deviation from Mean Rating", 
       x = "User Bias", y = "Number of Ratings")

# Print the user's bias mean and standard deviation.
mean(user_bias$user_bias)
sd(user_bias$user_bias)



##### Simple Baseline Approach #####
# To get a baseline, lets assume that all movies are rated the same with some random variance.

# Get train and test data sets
set.seed(100, sample.kind="Rounding")
trainIndex <- createDataPartition(edx$rating, p=0.8, list=FALSE)
train <- edx[trainIndex,]
test <- edx[-trainIndex,]

# Overall movie average rating
avgRating = mean(train$rating)

# Then we'll get our baseline RMSE value to compare future models against.
simpleRmse <- RMSE(test$rating, avgRating)

simpleRmse # 1.060809



##### Movie Effect #####
# Similar movies are often rated similarly. Add in a movie bias effect.

# Get train and test data sets
set.seed(200, sample.kind="Rounding")
trainIndex <- createDataPartition(edx$rating, p=0.8, list=FALSE)
train <- edx[trainIndex,]
test <- edx[-trainIndex,]

# Overall movie average rating
avgRating = mean(train$rating)

# Get each movie's average rating delta from overall movie averages
movieAvg <- train %>%
  group_by(movieId) %>%
  summarize(movieBias = mean(rating - avgRating))

# Use the movie bias to further develop the prediction model
predictedRating <- avgRating + test %>%
  left_join(movieAvg, by="movieId") %>%
  pull(movieBias)

movieEffectRmse <- RMSE(test$rating, predictedRating)

movieEffectRmse # 0.9446603



##### User Effect #####
# Users will also favor certain movies. Add a user effect term.

# Get train and test data sets
set.seed(300, sample.kind="Rounding")
trainIndex <- createDataPartition(edx$rating, p=0.8, list=FALSE)
train <- edx[trainIndex,]
test <- edx[-trainIndex,]

# Overall movie average rating
avgRating = mean(train$rating)

# Get each movie's average rating delta from overall movie averages
movieAvg <- train %>%
  group_by(movieId) %>%
  summarize(movieBias = mean(rating - avgRating))

# Get each user's average rating delta from overall movie averages
userAvg <- train %>%
  group_by(userId) %>%
  summarize(userBias = mean(rating - avgRating))

# Use the movie and user biase to develop the prediction model.
predictedRating <- test %>%
  left_join(userAvg, by="userId") %>%
  left_join(movieAvg, by="movieId") %>%
  mutate(pred = avgRating + userBias + movieBias) %>%
  pull(pred)

userEffectRmse <- RMSE(test$rating, predictedRating)

userEffectRmse # 0.8852448



##### Regularization #####
# Movies with a low amount of ratings will cause a larger variance in accuracy.
# Adjust the model to account for these variances using regularization.

# Get train and test data sets
set.seed(400, sample.kind="Rounding")
trainIndex <- createDataPartition(edx$rating, p=0.8, list=FALSE)
train <- edx[trainIndex,]
test <- edx[-trainIndex,]

# Overall movie average rating
avgRating = mean(train$rating)
# Minimize the penalized least squares equation including movie and user effects.
lambdas <- seq(3, 7, 0.25)

# Loop through each lambda to find the best value for this tuning parameter
# that minimizes the RMSE value when utilizing penalizing terms
rmses <- sapply(lambdas, function(lambda) {
  # Movie Effect
  b_movie <- train %>%
    group_by(movieId) %>%
    summarize(b_m = sum(rating - avgRating) / (n() + lambda))

  # User Effect
  b_user <- train %>%
    left_join(b_movie, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_m - avgRating) / (n() + lambda))

  # Prediction
  predicted_ratings <- test %>%
    left_join(b_movie, by="movieId") %>%
    left_join(b_user, by="userId") %>%
    mutate(prediction = avgRating + b_u + b_m) %>%
    pull(prediction)

  return(RMSE(predicted_ratings, test$rating))
})

# Print the best RMSE value found and associated lambda
bestLambda <- lambdas[which.min(rmses)]
regularizedRmse <- rmses[which.min(rmses)]

bestLambda # 5
regularizedRmse # 0.8650411

# Visualize each lambda's associated RMSE values to ensure a local minima was chosen
data.frame(lambdas, rmses) %>%
  ggplot(aes(lambdas, rmses)) +
  geom_point() +
  labs(title = "RMSE vs Lambda", x = "Lambda", y = "RMSE")


##### Final Prediction #####
# Movie Effect
b_movie <- edx %>%
group_by(movieId) %>%
summarize(b_m = sum(rating - avgRating) / (n() + bestLambda))

# User Effect
b_user <- edx %>%
left_join(b_movie, by="movieId") %>%
group_by(userId) %>%
summarize(b_u = sum(rating - b_m - avgRating) / (n() + bestLambda))

# Prediction
predicted_ratings <- final_holdout_test %>%
left_join(b_movie, by="movieId") %>%
left_join(b_user, by="userId") %>%
mutate(prediction = avgRating + b_u + b_m) %>%
pull(prediction)

finalRmse <- RMSE(predicted_ratings, final_holdout_test$rating)

finalRmse # 0.8648178





