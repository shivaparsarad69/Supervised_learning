
import pandas as pd
import numpy as np
from collections import Counter
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import collections

# preproccessing

def preprocessing(data):
    col_name = data.columns

    def miss_val(df,col):
        z = Counter(list(df[col]))
        sort = sorted(z.items(), key=lambda x: x[1], reverse=True)
        max_iteration = sort[0][0]
        df[col].replace('Nan', np.nan, inplace=True)
        df[col] = df[col].fillna(max_iteration)
        return df[col]
    def dummy_var(col,data):
        just_dummies = pd.get_dummies(data[col])
        print col
        print just_dummies

        step_1 = pd.concat([data, just_dummies], axis=1)
        step_1.drop([col,just_dummies.columns[0]], inplace=True, axis=1)
        return step_1

    data["Date"]=pd.to_datetime(data.Date)
    data["Start_time"]=pd.to_datetime(data.Start_time)
    data["End_time"]=pd.to_datetime(data.End_time)

    for i in range(data.shape[0]):

        if pd.isnull(data.loc[i]["Start_time"]):
            data.loc[i,"Start_time"]=data.loc[i,"Date"]
        if pd.isna(data.loc[i]["End_time"]):
            data.loc[i,"End_time"]=data.loc[i,"Date"]


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

    for col in data.columns:
        data[col]=miss_val(data,col)
    print "yes"
    # just for numinal variable
    for col in col_name :
        if col not in ("Date","End_time","Start_time","Length","Temperature in Montreal during episode"):
            data=dummy_var(col,data)
    return data
# End of preproccessing


def autoencoder(x):
    classifier = Sequential()
    input=classifier.add(Dense(units = x, kernel_initializer = 'uniform', activation = 'relu', input_dim = data.shape[1]))
    encoder=classifier.add(Dense(units = data.shape[1], kernel_initializer = 'uniform', activation = 'relu'))
    decoder=classifier.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics = ['accuracy'])
    classifier.fit(data, label, epochs = 100, batch_size = 10)
    return encoder


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
    return list(selected_features_BE)


# models
def neural_network(x,y,data,test_data,label):
    classifier = Sequential()
    classifier.add(Dense(units = x, kernel_initializer = 'uniform', activation = 'relu', input_dim = data.shape[1]))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = y, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics = ['accuracy'])
    classifier.fit(data, label, epochs = 100, batch_size = 10)
    y_pred=classifier.predict(test_data)
    print y_pred



data=pd.read_csv("/Users/shiva/Desktop/data.csv")
data=data.drop("Unnamed: 0",axis=1)

test_data=pd.read_csv("/Users/shiva/Desktop/test.csv")

#test_data=test_data.sample(frac=0.01,replace=False).reset_index(drop=True)

test_data=preprocessing(test_data)

#data=data.sample(frac=0.01, replace=False).reset_index(drop=True)
label=data["Market Share_total"]
data= data.drop('Market Share_total', axis=1)


new_data=preprocessing(data)

x_train, x_test, y_train, y_test = train_test_split(new_data, label, test_size=0.2)

lr = LinearRegression()
lr.fit(x_train, y_train)
lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(lr,data,label, cv=10)
print scores.mean()

ypred=neural_network(new_data.shape[1]+32,new_data.shape[1]+16,x_train,x_test)

final=np.where(ypred==y_test,"correct","incorrect")
counting=final.count("correct")
accuracy=counting/float(len(ypred))
print "Accuracy of neural network is : " + str(accuracy)

# for test set

result= neural_network(new_data.shape[1]+32,new_data.shape[1]+16,new_data,test_data,label)
print result
# in result we have no comparision because target column is absent in test data


# train and test neural network by new decreased feature space
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

