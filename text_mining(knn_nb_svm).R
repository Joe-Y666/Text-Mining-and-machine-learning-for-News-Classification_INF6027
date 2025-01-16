##predictive modelling
#################################################################################
install.packages('tm')
install.packages('SnowballC')
install.packages('glmnet')
install.packages('caret')
install.packages('e1071')
install.packages('class')
install.packages('naivebayes')
install.packages('lava')
install.packages('rsample')

library(tidyverse)
library(tm)
library(glmnet)
library(caret)
library(tm)
library(e1071)
library(class)
library(naivebayes)
library(rsample)

#load data
newsdata<-read_csv("/Users/joe/Downloads/MN-DS-news-classification.csv")
class(newsdata$content)

#Remove any rows containing missing values
newsdata <- na.omit(newsdata)

#Create a text corpus
news_data <- Corpus(VectorSource(newsdata$content))

#Preprocess text data (convert to lower, remove punctuation, numbers, stopwords and redundant space)
news_data <-tm_map(news_data, tolower)
news_data <-tm_map(news_data, removePunctuation)
news_data <-tm_map(news_data, removeNumbers)
news_data <-tm_map(news_data, removeWords, stopwords("en"))
news_data <-tm_map(news_data, stripWhitespace)

#Create a Document-Term Matrix (TF-IDF weighting)
news_dtm <- DocumentTermMatrix(news_data,control = list(weighting = weightTfIdf))
inspect(news_dtm[1:3,])

#The following command will keep the words in the top 98% of frequencies, removing all terms in the bottom 5% of frequencies
news_dtm<-removeSparseTerms(news_dtm, 0.98)
news_dtm_matrix <- as.matrix(news_dtm)

#Stratified Sampling
newsdata$category_level_1 <- factor(newsdata$category_level_1)
split <- initial_split(newsdata, prop = 0.7, strata = "category_level_1")
set.seed(42)
trainIndex <- createDataPartition(newsdata$category_level_1, p = 0.7, list = FALSE)
news_train_data <- news_dtm_matrix[trainIndex, ]
news_test_data <- news_dtm_matrix[-trainIndex, ]
news_train_labels <- newsdata$category_level_1[trainIndex]
news_test_labels <- newsdata$category_level_1[-trainIndex]



#train model
# Train and evaluate K-Nearest Neighbors model
knn_model <- knn(train = news_train_data, test = news_test_data, cl = news_train_labels, k = 5)
knn_pred <- knn_model
knn_cm <- confusionMatrix(knn_model, news_test_labels)
knn_results <- knn_cm$byClass
print(svm_cm)

# Train and evaluate Naive Bayes model
nb_model <- naiveBayes(news_train_data, news_train_labels)
nb_pred <- predict(nb_model, news_test_data)
nb_cm <- confusionMatrix(nb_pred, news_test_labels)
nb_results <- nb_cm$byClass
print(nb_cm)

# Train and evaluate SVM model
svm_model <- svm(news_train_data, as.factor(news_train_labels))
svm_pred <- predict(svm_model, news_test_data)
svm_cm <- confusionMatrix(svm_pred, news_test_labels)
svm_results <- svm_cm$byClass
print(svm_cm)

#Print evaluation metrics
list(KNN = knn_results, Naive_Bayes = nb_results, SVM = svm_results)

roc_curve <- roc(true_labels, predictions)
plot(roc_curve, main = "ROC Curve", col = "blue")

print(svm_cm$table)


#The result of classification
knn_results1 <- knn_cm$table
nb_results1 <- nb_cm$table
svm_results1 <- svm_cm$table
category_level_1 <- rownames(knn_results)

result_df <- data.frame(
  Category = category_level_1, 
  Number_of_Test_set = category_counts, 
  Classifier_in_KNN = diag(knn_results1), 
  Classifier_in_Naive_Bayes = diag(nb_results1), 
  Classifier_in_SVM = diag(svm_results1)
)
print(result_df)

#Export the classification results table
install.packages("writexl")
library(writexl)
write_xlsx(result_df, path = "/Users/joe/Downloads/result_df.xlsx")

#Performance evaluation
knn_precision <- knn_cm$byClass[,'Precision']
knn_recall <- knn_cm$byClass[,'Recall']
knn_f1 <- knn_cm$byClass[,'F1']

nb_precision <- nb_cm$byClass[,'Precision']
nb_recall <- nb_cm$byClass[,'Recall']
nb_f1 <- nb_cm$byClass[,'F1']

svm_precision <- svm_cm$byClass[,'Precision']
svm_recall <- svm_cm$byClass[,'Recall']
svm_f1 <- svm_cm$byClass[,'F1']

knn_precision_mean <- mean(knn_precision)
knn_recall_mean <- mean(knn_recall)
knn_f1_mean <- mean(knn_f1)

nb_precision_mean <- mean(nb_precision)
nb_recall_mean <- mean(nb_recall)
nb_f1_mean <- mean(nb_f1)

svm_precision_mean <- mean(svm_precision)
svm_recall_mean <- mean(svm_recall)
svm_f1_mean <- mean(svm_f1)

evaluation_df <- data.frame(
  Model = c('K-nearest neighbor', 'Naive Bayesian', 'SVM'), 
  Precision = c(knn_precision_mean, nb_precision_mean, svm_precision_mean),
  Recall = c(knn_recall_mean, nb_recall_mean, svm_recall_mean ), 
  F_value = c(knn_f1_mean, nb_f1_mean, svm_f1_mean )
)
write_xlsx(evaluation_df, path = "/Users/joe/Downloads/evaluation_df.xlsx")
