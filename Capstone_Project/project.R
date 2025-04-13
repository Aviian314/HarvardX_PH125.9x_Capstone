##### Imports #####
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(caret)) install.packages("caret")
if(!require(gridExtra)) install.packages("gridExtra")
if(!require(doParallel)) install.packages("doParallel")

library(tidyverse)
library(caret)
library(gridExtra)
library(doParallel)



##### Setup Parallel Processing #####
# Parallel processing speeds up the model training
num_cores <- detectCores() - 1  # Leave one core free for other tasks

# Register the parallel backend
cl <- makeCluster(num_cores)
registerDoParallel(cl)



##### Load data files #####
# Data from https://www.kaggle.com/datasets/uciml/mushroom-classification
#
# Attribute Information:
# class: edible=e, poisonous=p)
# cap-shape: bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s
# cap-surface: fibrous=f, grooves=g, scaly=y, smooth=s
# cap-color: brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u,
#            red=e, white=w, yellow=y
# bruises: bruises=t, no=f
# odor: almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n,
#       pungent=p, spicy=s
# gill-attachment: attached=a, descending=d, free=f, notched=n
# gill-spacing: close=c, crowded=w, distant=d
# gill-size: broad=b, narrow=n
# gill-color: black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o,
#             pink=p, purple=u, red=e, white=w, yellow=y
# stalk-shape: enlarging=e, tapering=t
# stalk-root: bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=?
# stalk-surface-above-ring: fibrous=f, scaly=y, silky=k, smooth=s
# stalk-surface-below-ring: fibrous=f, scaly=y, silky=k, smooth=s
# stalk-color-above-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o,
#                         pink=p, red=e, white=w, yellow=y
# stalk-color-below-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p,
#                         red=e, white=w, yellow=y
# veil-type: partial=p, universal=u
# veil-color: brown=n, orange=o, white=w, yellow=y
# ring-number: none=n, one=o, two=t
# ring-type: cobwebby=c, evanescent=e, flaring=f, large=l, none=n, pendant=p,
#            sheathing=s, zone=z
# spore-print-color: black=k, brown=n, buff=b, chocolate=h, green=r, orange=o,
#                    purple=u, white=w, yellow=y
# population: abundant=a, clustered=c, numerous=n, scattered=s, several=v, solitary=y
# habitat: grasses=g, leaves=l, meadows=m, paths=p, urban=u, waste=w, woods=d

mushroom_file <- "mushrooms.csv"
mushroom_all_data <- as.data.frame(
  read_csv(mushroom_file, col_names = TRUE)) %>% 
  select(-"veil-type") %>%
  mutate(across(everything(), as.factor))

# Faced issues with a column named "class" since this is a reserved key word
# Changing it to "class_label" to simplify processing
names(mushroom_all_data)[names(mushroom_all_data) == "class"] <- "class_label"



##### Final holdout test set #####
# Split data to get a final holdout test set
# This set will only be used to test completed models
set.seed(987)
splitInd <- createDataPartition(mushroom_all_data$class, p=0.7, list=FALSE)
mushroom_data <- mushroom_all_data[splitInd, ]
final_test_data <- mushroom_all_data[-splitInd, ]
rm(mushroom_all_data)



##### Data Cleanup. #####
# This dataset was already cleaned in preparation for machine learning
# so the expectation is that there's nothing to do.
mushroom_data_has_nulls <- any(is.na(mushroom_data))
final_test_data_has_nulls <- any(is.na(final_test_data))

mushroom_data_has_nulls
final_test_data_has_nulls



##### Data Exploration #####
# Plot all variables to see if any noticeable patterns appear.
long_data <- pivot_longer(mushroom_data, cols = everything(),
                          names_to = "variable", values_to = "value")

plots <- lapply(unique(long_data$variable), function(feature) {
  mushroom_data %>%
  select(all_of(feature), class_label) %>%
  ggplot(aes(x = !!sym(feature), fill = class_label)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(
    values = c("e" = "steelblue", "p" = "darkred"),
    labels = c("e" = "edible", "p" = "poisonous")
  ) +
  labs(title = feature, x = feature, y = "Count")
})

# Arrange the plots in a grid layout
grid.arrange(grobs = plots, ncol = 4)

# Odor stood out the most for uniqueness between classifications.
mushroom_data %>%
  select(odor, class_label) %>%
  ggplot(aes(x=odor, fill = class_label)) +
  geom_bar(position="dodge") +
  scale_fill_manual(
    values = c("e" = "steelblue", "p" = "darkred"),
    labels = c("e" = "edible", "p" = "poisonous")
  ) +
  labs(title = "Mushroom Classification by Odor", x = "Odor", y = "Count")



##### Simple Model #####
# This model uses a simple if/else statement to determine classification.
y_hat_simple <- factor(
  ifelse(mushroom_data$odor %in% c("a", "l", "n"), "e", "p"),
  levels = levels(mushroom_data$class_label))
cm_simple <- confusionMatrix(y_hat_simple, mushroom_data$class_label)
cm_simple # Balanced Accuracy: 0.9861



##### GLM Model #####
set.seed(123)
trainIndex <- createDataPartition(mushroom_data$class_label, p=0.8, list=FALSE)
train_data <- mushroom_data[trainIndex, ]
test_data <- mushroom_data[-trainIndex, ]

fit_glm <- train(class_label ~ ., data = train_data, method = "glm",
                 trControl = trainControl(method = "cv"))
y_hat_glm <- predict(fit_glm, newdata = test_data)
cm_glm <- confusionMatrix(y_hat_glm, test_data$class_label)
cm_glm # Balanced Accuracy: 1.000



##### KNN Model #####
set.seed(456)
trainIndex <- createDataPartition(mushroom_data$class_label, p=0.8, list=FALSE)
train_data <- mushroom_data[trainIndex, ]
test_data <- mushroom_data[-trainIndex, ]

fit_knn <- train(class_label ~ ., data = train_data, method = "knn",
                 trControl = trainControl(method = "cv"))
y_hat_knn <- predict(fit_knn, newdata = test_data)
cm_knn <- confusionMatrix(y_hat_knn, test_data$class_label)
cm_knn # Balanced Accuracy: 1.000



##### Random Forest Model #####
set.seed(789)
trainIndex <- createDataPartition(mushroom_data$class_label, p=0.8, list=FALSE)
train_data <- mushroom_data[trainIndex, ]
test_data <- mushroom_data[-trainIndex, ]

fit_rf <- train(class_label ~ ., data = train_data, method = "rf",
                trControl = trainControl(method = "cv"))
y_hat_rf <- predict(fit_rf, newdata = test_data)
cm_rf <- confusionMatrix(y_hat_rf, test_data$class_label)
cm_rf # Balanced Accuracy: 0.9991



##### final holdout test #####
y_hat_final_glm <- predict(fit_glm, newdata = final_test_data)
cm_final_glm <- confusionMatrix(y_hat_final_glm, final_test_data$class_label)
cm_final_glm # Balanced Accuracy: 1.000

y_hat_final_knn <- predict(fit_knn, newdata = final_test_data)
cm_final_knn <- confusionMatrix(y_hat_final_knn, final_test_data$class_label)
cm_final_knn # Balanced Accuracy: 0.9979

y_hat_final_rf <- predict(fit_rf, newdata = final_test_data)
cm_final_rf <- confusionMatrix(y_hat_final_rf, final_test_data$class_label)
cm_final_rf # Balanced Accuracy: 0.9991



##### Variable Importance #####
var_imp_glm <- varImp(fit_glm)
var_imp_knn <- varImp(fit_knn)
var_imp_rf <- varImp(fit_rf)

glm_imp <- var_imp_glm$importance %>% 
  rownames_to_column("Variable") %>%
  arrange(desc(Overall)) %>%
  slice(1:5) %>%
  mutate(Model = "GLM")

knn_imp <- var_imp_knn$importance %>% 
  rownames_to_column("Variable") %>%
  select(Variable, Overall = 2) %>%
  arrange(desc(Overall)) %>%
  slice(1:5) %>%
  mutate(Model = "KNN")

rf_imp <- var_imp_rf$importance %>% 
  rownames_to_column("Variable") %>%
  arrange(desc(Overall)) %>%
  slice(1:5) %>%
  mutate(Model = "Random Forest")

combined_imp <- bind_rows(glm_imp, knn_imp, rf_imp)

ggplot(combined_imp, aes(x = reorder(Variable, Overall),
                         y = Overall, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_flip() +
  labs(
    title = "Top 5 Most Important Variables per Model",
    x = "Variable",
    y = "Importance Score"
  ) +
  scale_fill_manual(values = c("GLM" = "steelblue",
                               "KNN" = "orange",
                               "Random Forest" = "darkgreen"))



##### Jack O' Lantern Mushroom Prediction #####
# Sources used to best populate the below data frame.
# https://www.mushroomexpert.com/omphalotus_olearius.html
# https://www.mushroom-appreciation.com/omphalotus-olearius.html
# https://en.wikipedia.org/wiki/Omphalotus_olearius

jack_o_lantern = data.frame(
  `class_label`="p",
  `cap-shape` = `"c",
  `cap-surface` = `"s",
  `cap-color` = `"y",
  `bruises` = `"FALSE",
  `odor` = `"n",
  `gill-attachment` = `"a",
  `gill-spacing` = `"c",
  `gill-size` = `"b",
  `gill-color` = `"y",
  `stalk-shape` = `"t",
  `stalk-root` = `"c",
  `stalk-surface-above-ring` = `"s",
  `stalk-surface-below-ring` = `"s",
  `stalk-color-above-ring` = `"o",
  `stalk-color-below-ring` = `"o",
  `veil-color` = `"o",
  `ring-number` = `"o",
  `ring-type` = `"e",
  `spore-print-color` = `"w",
  `population` = `"c",
  `habitat` = `"d",
  check.names = FALSE
)

# Predict the mushroom class using the models.
# simple model: odor "n" => edible
predict(fit_glm, newdata=jack_o_lantern) # edible
predict(fit_knn, newdata=jack_o_lantern) # edible
predict(fit_rf, newdata=jack_o_lantern) # edible



##### Non-Generalization Example #####
# To further elaborate on the jack-o-lantern example
# Train a model using forest mushrooms and then 
# predict the grass mushroom's class.
forest_mushrooms <- mushroom_data %>% filter(habitat == "u")
grass_mushrooms <- mushroom_data %>% filter(habitat == "g")

# Train on forest, test on grass
fit_forest <- train(class_label ~ ., data = forest_mushrooms, method = "glm",
                    trControl = trainControl(method = "cv"))
y_hat_grass <- predict(fit_forest, newdata = grass_mushrooms)
cm_grass <- confusionMatrix(y_hat_grass, grass_mushrooms$class_label)
cm_grass # Balanced Accuracy: 0.5299

# False Positive Rate
FP <- cm_grass$table["e", "p"]
TN <- cm_grass$table["p", "p"]
FPR <- FP / (FP + TN)
FPR



##### Stop Cluster #####
stopCluster(cl)
