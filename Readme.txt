Group members: Mengyun Lv & Yuetong Zhao
course project for EECS 510

Before running the code, please make sure the library NLTK is installed, which will be used for lemmatization in data processing. 

The version of python we use is 2.7.6, some functions may not be supported in earlier version.  
1. Make sure that the train and test data are in the Data directory
2.  run “./clean_data.py" : run clean_data for query normalization and time analysis, and the new training and test set will be generated in the data directory named clean_test.csv and clean_train.csv
3.  run "./train_predict.py ../data/clean_train.csv ../data/clean_test.csv": run train_predict to train the model and generate prediction of 5 best skus for the each test line, and the results will be generated in the data directory named prediction.csv, which is in the same syntax as the benchmark file popular_skus.csv
