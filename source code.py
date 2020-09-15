#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense , Dropout , LSTM
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


#read the file
df = pd.read_csv('D:/NSE-TATAGLOBAL11.csv')
#setting index as date
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']
# In[3]:
plt.figure(figsize=(16,8))
plt.plot(df['Close'], label='Close Price history')
plt.legend()
df.head()


# In[3]:


#creating dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]


new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)


# In[4]:


#creating train and test sets
dataset = new_data.values

train = dataset[0:987,:]
valid = dataset[987:,:]
len(dataset)
print(train[0:10])


# In[5]:


len(dataset)


# In[6]:


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
print(scaled_data)


# In[7]:


x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)
print(x_train)
#print(np.array(x_train.shape[0]))
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
#print(x_train)
print(x_train.shape)


# In[8]:


print(y_train.shape)
print(x_train.shape)


# In[9]:


print(len(y_train))
print(len(x_train))


# In[10]:


input_shape=(x_train.shape[1],1)
print(input_shape)


# In[11]:


model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))


# In[12]:


model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)


# In[13]:


#predicting 246 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)


# In[14]:


X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)


# In[15]:


X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)


# In[16]:


rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
print(rms)


# In[17]:


#for plotting
plt.figure(figsize=(16,8))
train = new_data[:987]
valid = new_data[987:]
valid['Predictions'] = closing_price
plt.plot(train['Close'],label='training_data')
plt.plot(valid['Close'],label=['testing_data'])
plt.plot(valid['Predictions' , labl])
plt.legend()


# In[ ]:




