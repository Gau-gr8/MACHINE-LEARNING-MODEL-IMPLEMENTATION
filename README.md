# MACHINE-LEARNING-MODEL-IMPLEMENTATION

*COMPANY*: CODETECH IT SOLUTIONS

*NAME*: GAURI AGARWAL

*INTERN ID*: CT06DF709

*DOMAIN*: PYTHON PROGRAMMING

*DURATION*: 6 WEEKS

*MENTOR*: NEELA SANTHOSH
                                          
*DESCRIPTION*:
The task of spam email classification is a well-known machine learning problem that involves building a predictive model to identify whether a given message is spam or ham. In this project, I implemented a spam classifier using Python, the Scikit-learn library, and a Google Colab notebook. The dataset used for this task was downloaded from the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/228/sms+spam+collection), which provides a widely recognized "SMS Spam Collection Dataset".

This dataset consists of over 5,000 SMS messages labeled as either ‘spam’ or ‘ham’. Each row contains a label and a message. The objective was to train a machine learning model that could accurately classify future messages based on the patterns learned from the data.

I began the task by uploading the dataset into Google Colab using the files.upload() utility. Since the dataset is tab-separated (not comma-separated), I used the sep='\t' parameter while loading the file using Pandas. I also assigned column names ['label', 'message'] manually, as the dataset doesn't contain a header row.

Before building the model, it was important to clean the data. Using Pandas, I dropped any missing or malformed rows to prevent errors during model training. Then, I converted the label column into numerical format, mapping 'ham' to 0 and 'spam' to 1. The message column was also cleaned to ensure that there were no NaN values.

Once the dataset was ready, I split it into training and testing sets using train_test_split() from Scikit-learn. To convert the raw text messages into numerical features, I used CountVectorizer(), a bag-of-words method that tokenizes the text and builds a matrix of word counts. This allowed the machine learning algorithm to process the text numerically.

For model training, I chose the Multinomial Naive Bayes classifier, which is particularly effective for text classification tasks. The model was trained using the training set and then tested on the unseen data (the test set). I used various evaluation metrics from Scikit-learn, including accuracy score, classification report, and a confusion matrix, to assess the model's performance.

The output showed that the classifier achieved very high accuracy which demonstrates that, it is highly effective when paired with proper preprocessing and a suitable dataset.

All of this work was done in Google Colab, a cloud-based Jupyter notebook environment provided by Google. Colab is particularly helpful because it allows for easy execution of Python code in the browser, supports GPU acceleration, and simplifies file uploads and sharing. I also found it ideal for experimenting and visualizing results in real-time without needing to install any software locally.

*TOOLS AND LIBRARIES USED:*
1.GOOGLE COLAB: Online notebook environment for coding, testing, and documentation

2. PANDAS: Data handling and preprocessing

3.SCIKIT-LEARN: Model building and evaluation

4.MATPLOTLIB AND SEABORN: Data visualization

5.UCI ML Repository: Source of the dataset

*APPLICATIONS OF THIS TASK:*
This type of classification model can be used in real-world systems such as:

1.Email filtering (Gmail, Outlook)

2.SMS spam detection

3.Customer service automation

4.Content moderation

5.Fraud and phishing detection

*OUTPUTS:*
![Image](https://github.com/user-attachments/assets/2c261e1e-7783-4fcb-9563-a578f569efda)

![Image](https://github.com/user-attachments/assets/312595d2-fd8c-4705-8b8e-c4c6dc3ac5f1)
