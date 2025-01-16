#Install and acsess the required packages
install.packages("wordcloud")
install.packages("tm")        
install.packages("RColorBrewer")
library(wordcloud)
library(tm)
library(RColorBrewer)


#Exploratory analysis with text mining

#load data
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

#Frequency and correlation
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

#wordcloud
wordcloud_dtm<- TermDocumentMatrix(sport_news_docs)
wordcloud_dtm_matrix <- as.matrix(wordcloud_dtm)
wordcloud_dtm_freq <- sort(rowSums(wordcloud_dtm_matrix), decreasing = TRUE)
wordcloud_data <- data.frame(word = names(wordcloud_dtm_freq), freq = wordcloud_dtm_freq)
set.seed(1234)

# Different word cloud styles
#wordcloud(words = wordcloud_data$word, freq = wordcloud_data$freq, 
          min.freq = 50,      
          max.words = 100,  
          random.order = FALSE,
          colors = brewer.pal(8, "Dark2"))

wordcloud(words = wordcloud_data$word, freq = wordcloud_data$freq,
          min.freq = 2, scale = c(3, 0.5), colors = brewer.pal(6, "Set2"),
          random.order = FALSE, rot.per = 0.35, use.r.layout = FALSE)
