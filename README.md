# Text-Mining-and-machine-learning-for-News-Classification_INF6027



## Author Introduction
Hello! I'm Joe, a student in UK. a data science enthusiast focusing on data analysis and machine learning using R.

- **GitHub**: [Author's GitHub link ](https://github.com/Joe-Y666)

|    Author  | Joe        |
|------------|------------|
| Hobby      | Travel, hot pot, sports   |
| professional skill | R, python, SPSS, SQL  |

## Project Title
Text Mining and machine learning for News Classification(INF6027)

## Project Description
This study used Multilabeled News Dataset(MN-DS) from internet. The distribution of each will be analyzed <br>after process missing value in news sample. Then, the study started explore the frequency of term and correlation between words.
Text preprocessing was carried out by using lower case, removing punctuation and stopping words in text classification.<br> Then, TF-IDF was used for feature extraction. Compare the performance of three machine learning models, including K-nearest neighbor algorithm (KNN), Naive Bayes algorithm (NB), and support vector machine (SVM), evaluating the precision, recall, and F1. All analysis in this study used R studio. 

## Reseach Question
- What is the association between content and category?
- Which algorithms can provide better accuracy and effect in the multi-category news classification task？


## Rcode and comment

### Exploratory analysis with text mining




```r
#Before using this analysis, please make sure to install the following R packages:
install.packages("wordcloud")
install.packages("tm")        
install.packages("RColorBrewer")

library(wordcloud)
library(tm)
library(RColorBrewer)

# load data
data<-read_csv("/Users/joe/Downloads/MN-DS-news-classification.csv")
class(newsdata$content)
data <- na.omit(data)
sport_news <- filter(data, category_level_1 == "sport")
sport_news_docs <-Corpus(VectorSource(sport_news$content))
wordcloud <- c(stopwords("en"), "will", "also", "thus", "however", "like", "one", "just", "get", "can", "'s", "said", "much", "make", "made","now","will")
sport_news_docs <- tm_map(sport_news_docs, tolower)
sport_news_docs <- tm_map(sport_news_docs, removePunctuation)
sport_news_docs <- tm_map(sport_news_docs, removeNumbers)
sport_news_docs <- tm_map(sport_news_docs, removeWords, wordcloud)
sport_news_docs <- tm_map(sport_news_docs, stripWhitespace)

# Frequency and correlation
sportnews_dtm <- DocumentTermMatrix(sport_news_docs)
findFreqTerms(sportnews_dtm, lowfreq=100)
findAssocs(sportnews_dtm, 'football',
  corlimit=0.35 # the correlation limit (between 0 and 1)
)
findAssocs(sportnews_dtm, 'league',
  corlimit= 0.55 )  # the correlation limit (between 0 and 1)

findAssocs(sportnews_dtm, 'fulham',
  corlimit=0.75 # the correlation limit (between 0 and 1)
)

# wordcloud
wordcloud_dtm<- TermDocumentMatrix(sport_news_docs)
wordcloud_dtm_matrix <- as.matrix(wordcloud_dtm)
wordcloud_dtm_freq <- sort(rowSums(wordcloud_dtm_matrix), decreasing = TRUE)
wordcloud_data <- data.frame(word = names(wordcloud_dtm_freq), freq = wordcloud_dtm_freq)
set.seed(1234)
#Different styles of wordcloud
#wordcloud(words = wordcloud_data$word, freq = wordcloud_data$freq, 
          min.freq = 50,      
          max.words = 100,  
          random.order = FALSE,
          colors = brewer.pal(8, "Dark2"))

wordcloud(words = wordcloud_data$word, freq = wordcloud_data$freq,
          min.freq = 2, scale = c(3, 0.5), colors = brewer.pal(8, "Set2"),
          random.order = FALSE, rot.per = 0.35, use.r.layout = FALSE)



## Text Mining
## Installation Dependencies for text classification
```r

#Before using this analysis, please make sure to install the following R packages:
install.packages('tm')
install.packages('SnowballC')
install.packages('glmnet')
install.packages('caret')
install.packages('e1071')
install.packages('class')
install.packages('naivebayes')
install.packages('lava')

# In the following R code, we load the necessary libraries and read in the data.
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
set.seed(42) # allows you to reproduce random number generation
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

# KNN Performance evaluation
knn_precision <- knn_cm$byClass[,'Precision']
knn_recall <- knn_cm$byClass[,'Recall']
knn_f1 <- knn_cm$byClass[,'F1']
# NB Performance evaluation
nb_precision <- nb_cm$byClass[,'Precision']
nb_recall <- nb_cm$byClass[,'Recall']
nb_f1 <- nb_cm$byClass[,'F1']

# SVM Performance evaluation
svm_precision <- svm_cm$byClass[,'Precision']
svm_recall <- svm_cm$byClass[,'Recall']
svm_f1 <- svm_cm$byClass[,'F1']

#Index averaging
knn_precision_mean <- mean(knn_precision)
knn_recall_mean <- mean(knn_recall)
knn_f1_mean <- mean(knn_f1)

nb_precision_mean <- mean(nb_precision)
nb_recall_mean <- mean(nb_recall)
nb_f1_mean <- mean(nb_f1)

svm_precision_mean <- mean(svm_precision)
svm_recall_mean <- mean(svm_recall)
svm_f1_mean <- mean(svm_f1)

# Creat a datafram of result
evaluation_df <- data.frame(
  Model = c('K-nearest neighbor', 'Naive Bayesian', 'SVM'), 
  Precision = c(knn_precision_mean, nb_precision_mean, svm_precision_mean),
  Recall = c(knn_recall_mean, nb_recall_mean, svm_recall_mean ), 
  F_value = c(knn_f1_mean, nb_f1_mean, svm_f1_mean )
)
#Export results
write_xlsx(evaluation_df, path = "/Users/joe/Downloads/evaluation_df.xlsx")
