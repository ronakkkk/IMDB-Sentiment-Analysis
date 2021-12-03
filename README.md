# IMDB-Sentiment-Analysis
So, the data used here is the IMDB sentiments which includes movies reviews, and the sentiments of the reviews whether it is positive or negative.

Total data are 50000 from which 25000 data are used for training the model and 25000 data are used to test the model.

Initially, the data are not in the format which can be used for the processing of the model. So the data is converted into the particular format where the positive labels are changed with 0 and negative labels with 1 and align with the text reviews.

For the given data text column is the feature for the machine learning model and class is the label.

But the given feature is in the text format so have to change in the numeric feature so it can be processed by the given machine learning mode.

To convert the text data into numeric feature TF-IDF vectorizer is used which converts the data into numeric feature by taking consideration of the frequency of the word in the given texts.

Now the given data have numeric feature and have numeric label, so it can be processed and used to build machine learning model.

So, for the given data we try to use binary classification using Support vector machine model, in which we have used support vector classifier and random forest classifier model. Using two model is to find which model is good for the given dataset as each model works differently. 

SVC works on radial basis function kernel which are used in kernelized learning algorithm, through which our SVC model is learned on the given train data.

Therefore the model is trained and tested and using classification report and confusion matrix it can be said that support vector machine performs better as the data doesn't include any outliers which help to make clusters for the given sentiments and using hyperplane line it would make it  easy to predict where the given feature will fall in the given clusters.

Classification report tell us how much precise, total recall and what is the f1-score of our model. While confusion matrix tells us how much of the given data it is able  to predict right and how much it predict wrong which shows a clear idea how the accuracy of the model is decided.

