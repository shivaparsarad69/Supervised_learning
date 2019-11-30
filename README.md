# Supervised_learning
A supervised learning for predict market share

In this project i tried to write a supervised algorithm sunch as linear regression and neural network to predict market share. my target value, total market share, is continues valu, so some supervised algorithm such as ID3, Naive bayes,... are not good choices for this prediction.
We'll be using the popular data manipulation framework pandas. We're going to go ahead and load all of our dataset as a pandas Dataframe. "Unnamed: 0" is index variable and should be drop.


    data=pd.read_csv("/Users/shiva/Desktop/data.csv")
    data=data.drop("Unnamed: 0",axis=1)
 
 
about my features, some of them are nominal, that this is a chalenge for regression and neural network models, we should convert these nominal features to numerical type. and there is a problem with missing values too. there are different ways to deal this problem, replace missing value with most reapeated value on related column is common.

    def miss_val(df,col):
        z = Counter(list(df[col]))
        sort = sorted(z.items(), key=lambda x: x[1], reverse=True)
        max_iteration = sort[0][0]
        df[col].replace('Nan', np.nan, inplace=True)
        df[col] = df[col].fillna(max_iteration)
        return df[col]


but about data and time variable in this spacial data set, we went different path to deal with missing value problem. In this dataset "Date" variable and "End_time" and "Start_time" variables are datatime type (ofcourse we convert them to datatime type), imagine "Start_time" is None in a row, according to data, it is most reasonable to fill this missed value with "Data" content for this row. and same scenario is true for "End_time" variable.

    for i in range(data.shape[0]):

        if pd.isnull(data.loc[i]["Start_time"]):
            data.loc[i,"Start_time"]=data.loc[i,"Date"]
        if pd.isna(data.loc[i]["End_time"]):
            data.loc[i,"End_time"]=data.loc[i,"Date"]


about catogorical variables, dummy variable (https://stackoverflow.com/questions/11587782/creating-dummy-variables-in-pandas-for-python) that is a widely used method for encoding categorical variables to convert nominal variables is used.


    def dummy_var(col,data):
        just_dummies = pd.get_dummies(data[col])
        print col
        print just_dummies


but there is a problem with  "Date", "End time" and "Start time" features,they include data about date and time, and some data like month are cyclic, the proper handling of such features involves representing the cyclical features as (x,y) coordinates on a circle.We map each cyclical variable onto a circle such that the lowest value for that variable appears right next to the largest value. We compute the x- and y- component of that point using sin and cos trigonometric functions.In this representation hour, 23 and hour 0 are right next to each other numerically, just as they should be. but about hour, minute and second,first, i calculated total seconds and then compute x- and y- component using sin and cos, beacue i realized, computing cos and sin for different elements of a time could not provide additional information for prediction task. 

   
       def total_seconds(col):
         second=data[col].dt.second
         minute=data[col].dt.minute
         hour=data[col].dt.hour
         for i in range(len(second)):
             data["x"] = int(hour[i]) * 3600 + int(minute[i]) * 60 + int(second[i])
         return data["x"]

    seconds_in_day = 24*60*60
    data["Time_End"]=total_seconds("End_time")
    data["Time_Start"]=total_seconds("Start_time")
    data.drop('x', axis=1, inplace=True)


    data['sin_time_End'] = np.sin(2*np.pi*data.Time_End/seconds_in_day)
    data['cos_time_End'] = np.cos(2*np.pi*data.Time_End/seconds_in_day)
    data.drop("Time_End",axis=1,inplace=True)

    data['sin_time_Start'] = np.sin(2*np.pi*data.Time_Start/seconds_in_day)
    data['cos_time_Start'] = np.cos(2*np.pi*data.Time_Start/seconds_in_day)
    data.drop("Time_Start",axis=1,inplace=True)



    data['mnth_sin_End'] = np.sin((data.End_time.dt.month-1)*(2.*np.pi/12))
    data['mnth_cos_End'] = np.cos((data.End_time.dt.month-1)*(2.*np.pi/12))


    data['day_sin_End']=np.sin((2*np.pi)/30*data.End_time.dt.day)
    data['day_cos_End']=np.cos((2*np.pi)/30*data.End_time.dt.day)


    data['mnth_sin_Start'] = np.sin((data.Start_time.dt.month-1)*(2.*np.pi/12))
    data['mnth_cos_Start'] = np.cos((data.Start_time.dt.month-1)*(2.*np.pi/12))


    data['day_sin_Start']=np.sin((2*np.pi)/30*data.Start_time.dt.day)
    data['day_cos_Start']=np.cos((2*np.pi)/30*data.Start_time.dt.day)

    data.drop("Date",axis=1,inplace=True)
    data.drop("End_time",axis=1,inplace=True)
    data.drop("Start_time",axis=1,inplace=True)
    
According to the above code, after creating new features based on "Date" , "End_time" and "Start_time", these features are droped from dataframe.
We are able to do any supervised learning that input continues value, like neural network and regression, below is implemention of a simple neural network algorithm for this task.

    def neural_network(x,y):
  
      classifier = Sequential()
      classifier.add(Dense(units = x, kernel_initializer = 'uniform', activation = 'relu', input_dim = data.shape[1])  
      classifier.add(Dropout(0.2))
      classifier.add(Dense(units = y, kernel_initializer = 'uniform', activation = 'relu'))
      classifier.add(Dropout(0.2))
      classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'relu'))
      classifier.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics = ['accuracy'])
      classifier.fit(data, label, epochs = 100, batch_size = 10)
      y_pred=classifier.predict(test_data)
      print y_pred
      

    
  x,y are hidden layer dimensions, that determine experimentally.
  
  We can use regression model as below:
      
      lr = LinearRegression()
      lr.fit(x_train, y_train)
      lr_confidence = lr.score(x_test, y_test)
      print("lr confidence: ", lr_confidence)
      from sklearn.model_selection import cross_val_score
      scores = cross_val_score(lr,data,label, cv=10)
      print scores.mean()

lr confidence is R^2 measure. In this code we used 10 fold cross validation to evaluate our regression model.
but above codes are not optimized. and about calling neural network:

      ypred=neural_network(data.shape[1]+32,data.shape[1]+16,x_train,x_test)
      final=np.where(ypred==y_test,"correct","incorrect")
      counting=final.count("correct")
      accuracy=counting/float(len(ypred))
      print "Accuracy of neural network is : " + str(accuracy)
about test set, target column is absent so doing comparision and declare results in terms of for example accuracy is not possible. so we just report predictions.

      result= neural_network(data.shape[1]+32,data.shape[1]+16,new_data,test_data,label)
      print result

this data set has too many columns, and training time is very time consuming, so feature selection is essential. My suggestion is autoencoder or  Backward Elimination, in autoencoder we reach a new representation of features data and we used these new features as prediction variables. Output result of backward elimination is a subset of original features.
 
    def autoencoder(x):
      classifier = Sequential()
      input=classifier.add(Dense(units = x, kernel_initializer = 'uniform', activation = 'relu', input_dim = data.shape[1]))
      encoder=classifier.add(Dense(units = data.shape[1], kernel_initializer = 'uniform', activation = 'relu'))
      decoder=classifier.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics = ['accuracy'])
      classifier.fit(data, label, epochs = 100, batch_size = 10)
      return encoder
      
encoder is a new representation for our data. The final values in neurons of encoder layer is new representation of our data and could be replaced by our original space.
about backward elimination, the below code could be used:

    def backward_elimination(X,y):  # X is data, y is target variable
      cols = list(X.columns)
      pmax = 1
      while (len(cols)>0):
          p= []
          X_1 = X[cols]
          X_1 = sm.add_constant(X_1)
          model = sm.OLS(y,X_1).fit()
          p = pd.Series(model.pvalues.values[1:],index = cols)
          pmax = max(p)
          feature_with_p_max = p.idxmax()
          if(pmax>0.05):
              cols.remove(feature_with_p_max)
          else:
              break
      selected_features_BE = cols
      return selected_features_BE

then we could train our model with these selected features instead of total features space.

    new_feature_space = backward_elimination(new_data,label)
    new_data.drop(new_data.columns.difference([new_feature_space]), 1, inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(new_data, label, test_size=0.2)
    ypred2=neural_network(new_data.shape[1]+32,new_data.shape[1]+16,x_train,x_test)

    final=np.where(ypred2==y_test,"correct","incorrect")
    counting2=final.count("correct")
    accuracy2=counting2/float(len(ypred2))
    print "Accuracy of neural network with selected features is : " + str(accuracy2)

    test_data.drop(test_data.columns.difference([new_feature_space]), 1, inplace=True)
    result= neural_network(new_data.shape[1]+32,new_data.shape[1]+16,new_data,test_data,label)
    print result
    
    
Could we conclude that just regression and neural networks are good choices for this data set ? my answer is no, we could try CNN, and some other deep learning methods too (our dataset is big and deep learning needs too much data, so deep is a good choice, but it's training takes a lot of space and time). about LSTM, i'm not sure, beacuse i dont know whether our next instance is related to previous one or not, for example in stock market, we can use LSTM beacuse the next stock price depends on the previous one, but about this data i'm not sure.


# this code is not compelet and needs more reform and completion.

 






  
