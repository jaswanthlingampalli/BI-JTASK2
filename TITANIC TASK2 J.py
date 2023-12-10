#!/usr/bin/env python
# coding: utf-8

# # # ## BHARAT INTERNSHIP
# # # 
# # #   ## NAME-LINGAMPALLI JASWANTH
# # #   
# # #   ## TASK 2-TITANIC CLASSIFICATION
# # #   - In this we predicts if a passenger will survive on the titanic or not

# ## Importing needed libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Accuiring data

# In[2]:


train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
combine=[train_df, test_df]  #This helps us to run certain commands to the both datasets.


# ## Checking data and initial research
# 
# 

# In[3]:


train_df.head()


# #### Which features contain blank, null or empty values? 

# In[4]:


train_df.info()
print("-"*20)
test_df.info()


# **These will require correcting:** Age, Cabin and embarked in the training dataset. Age, Cabin and Fare in the test set. 

# #### What is the distribution of numerical feature values across the samples?
# 
# This helps us determine, among other early insights, how representative is the training dataset of the actual problem domain.
# 

# In[5]:


train_df.describe()


# #### What is the distribution of categorical features?

# In[6]:


train_df.describe(include=['O'])


# ### Initial assumptions based on data analysis so far

# ### Pclass

# 

# In[7]:


sb.catplot(x = "Survived", col = "Pclass", col_wrap = 3, data = train_df, kind = "count")
plt.show();


# ### Sex

# In[8]:


train_df[['Sex']].value_counts()


# In[9]:


women = train_df[train_df.Sex == 'female']['Survived']
men = train_df[train_df.Sex == 'male']['Survived']
print("From women " + str(round(women.sum()*100/women.count(),2)) + " % survived.")
print("From men " + str(round(men.sum()*100/men.count(),2)) + " % survived.")


# In[10]:


women_class1 = train_df[(train_df.Sex == 'female') & (train_df.Pclass == 1)]['Survived']
women_class2 = train_df[(train_df.Sex == 'female') & (train_df.Pclass == 2)]['Survived']
women_class3 = train_df[(train_df.Sex == 'female') & (train_df.Pclass == 3)]['Survived']

print( "In passenger class 1 where " + str(round(women_class1.count()*100/women.count(),2)) + " % of women.")
print( "In passenger class 2 where " + str(round(women_class2.count()*100/women.count(),2)) + " % of women.")
print( "In passenger class 3 where " + str(round(women_class3.count()*100/women.count(),2)) + " % of women.")
print("\n")
print("In passenger class 1 " + str(round(women_class1.sum()*100/women_class1.count(), 2)) + " % of its women survived."  )
print("In passenger class 2 " + str(round(women_class2.sum()*100/women_class2.count(), 2)) + " % of its women survived."  )
print("In passenger class 3 " + str(round(women_class3.sum()*100/women_class3.count(), 2)) + " % of its women survived."  )


# ### Age

# 
# The age information is missing from 177 passangers. I would think that age is significant contributor for surviving. Let see different age groups in more detail. 

# In[11]:


age_groups = []

for i in range(1,16):
  age_groups.append(str(i*5+1) + ' - ' + str(i*5+5) + ' years')
age_study = pd.DataFrame(index = age_groups)
age_study['Frequency'] = [train_df[(train_df.Age < (i*5+6)) & (train_df.Age >= (i*5+1))]['Age'].size for i in range(1,16)]
age_study['Frequency %'] = round(age_study['Frequency']*100/891, 2)
age_study['Survived'] = [train_df[(train_df.Age < (i*5+6)) & (train_df.Age >= (i*5+1))]['Survived'].sum() for i in range(1,16)]
age_study['Survived %'] = round(age_study['Survived']*100/age_study['Frequency'],2)

age_smallchilds = pd.DataFrame(index = ['0 - 5 years'])
age_smallchilds['Frequency'] = [train_df[train_df.Age < 6]['Age'].size]
age_smallchilds['Frequency %'] = round(age_smallchilds['Frequency']*100/891, 2)
age_smallchilds['Survived'] = [train_df[train_df.Age < 6]['Survived'].sum()]
age_smallchilds['Survived %'] = round(age_smallchilds['Survived']*100/age_smallchilds['Frequency'],2)

no_age = pd.DataFrame(index = ['No Age Info'])
no_age['Frequency'] = [train_df[train_df.Age.isna()]['Age'].size]
no_age['Frequency %'] = round(no_age['Frequency']*100/891, 2)
no_age['Survived'] = [train_df[train_df.Age.isna()]['Survived'].sum()]
no_age['Survived %'] = round(no_age['Survived']*100/no_age['Frequency'],2)

age_study = pd.concat([age_smallchilds, age_study, no_age])
age_study


# In[12]:


age_study[['Frequency']].sum() # Checking that all dataset entries got counted.


# ### SibSp (number of siblings or spouses aboard) 

# In[13]:


train_df[['SibSp']].value_counts()


# In[14]:


for i in range(9):
    alives = len(train_df[(train_df['SibSp'] == i) & (train_df['Survived'] == 1)])
    deads = len(train_df[(train_df['SibSp'] == i) & (train_df['Survived'] == 0)])
    if deads + alives != 0:
      survival_rate = round(alives*100/(alives + deads), 2)
    else:
      survival_rate = "No passengers with this SibSp"
    print(f"SibSp = {i}: Survived {alives}, Died {deads}, Survival rate : {survival_rate} %")


# This value seems also effecting the survival rate.

# ### Parch (number of parents or children aboard)
# So other number looking for the family size.

# In[15]:


train_df[['Parch']].value_counts()


# In[16]:


for i in range(7):
    alives = len(train_df[(train_df['Parch'] == i) & (train_df['Survived'] == 1)])
    deads = len(train_df[(train_df['Parch'] == i) & (train_df['Survived'] == 0)])
    if deads + alives != 0:
      survival_rate = round(alives*100/(alives + deads), 2)
    else:
      survival_rate = "No passengers with this Parch"
    print(f"Parch = {i}: Survived {alives}, Died {deads}, Survival rate : {survival_rate} %")


# It seems also effecting the surviving. Without parents or children you were more likely to die.

# ### Fare

# In[17]:


plt.hist(x = [train_df[train_df['Survived']==1]['Fare'], train_df[train_df['Survived']==0]['Fare']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'], bins = 100)
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare')
plt.ylabel('# of Passengers')
plt.legend();


# In[18]:


plt.hist(x = [train_df[train_df['Survived']==1]['Fare'], train_df[train_df['Survived']==0]['Fare']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'], bins = 100)
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare')
plt.ylabel('# of Passengers')
plt.xlim(left = 0, right = 50)
plt.legend();


# In[19]:


plt.hist(x = [train_df[train_df['Survived']==1]['Fare'], train_df[train_df['Survived']==0]['Fare']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'], bins = 100)
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare')
plt.ylabel('# of Passengers')
plt.xlim(left = 50, right = 100)
plt.ylim(top = 50)
plt.legend();


# In[20]:


plt.hist(x = [train_df[train_df['Survived']==1]['Fare'], train_df[train_df['Survived']==0]['Fare']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'], bins = 100)
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare')
plt.ylabel('# of Passengers')
plt.xlim(left = 100)
plt.ylim(top=30)
plt.legend();


# ### Cabin
# This in formation is available only for 204 passengers so I will ignore it. 

# ### Embarked
# C = Cherbourg, Q = Queenstown, S = Southampton

# In[21]:


sb.catplot(x = "Survived", col = "Embarked", col_wrap = 3, data = train_df, kind = "count")
plt.show();


# Seems that if you embarked from Southampton you were more likely to not survive. Is this because of the PClass?

# In[22]:


sb.catplot(x = "Pclass", col = "Embarked", col_wrap = 3, data = train_df, kind = "count")
plt.show();


# ## Preprocessing data

# In[23]:


# Combining SibSp and Parch
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp']+ dataset['Parch']+1 #One added because familysize 0 for singles sounds weird.


# In[24]:


# Drop out the 80-year old outflier
train_df.drop(index=train_df['Age'].idxmax(), inplace=True)
train_df['Age'].max()


# In[25]:


#3. Dropping out the three passengers with really hig fare. 
for i in range(3):
  train_df.drop(index=train_df['Fare'].idxmax(), inplace=True)
train_df['Fare'].max()


# In[26]:


#Filling missing values for Embarked
for dataset in combine:
  dataset["Embarked"] = dataset["Embarked"].fillna('S')


# In[27]:


#Making the Title feature
for dataset in combine:
  dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False) #This extracts from name the part that ends to the dot


# In[28]:


train_df['Title'].value_counts()


# In[29]:


#Let's simplify titles.
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


# In[30]:


train_df['Title'].value_counts()


# In[31]:


# Now we can fill the empty ages.
for dataset in combine:
  dataset.loc[(dataset.Age.isnull())&(dataset.Title=='Mr'),'Age']= dataset.Age[dataset.Title=="Mr"].mean()
  dataset.loc[(dataset.Age.isnull())&(dataset.Title=='Mrs'),'Age']= dataset.Age[dataset.Title=="Mrs"].mean()
  dataset.loc[(dataset.Age.isnull())&(dataset.Title=='Master'),'Age']= dataset.Age[dataset.Title=="Master"].mean()
  dataset.loc[(dataset.Age.isnull())&(dataset.Title=='Miss'),'Age']= dataset.Age[dataset.Title=="Miss"].mean()
  dataset.loc[(dataset.Age.isnull())&(dataset.Title=='Rare'),'Age']= dataset.Age[dataset.Title=="Rare"].mean()


# In[32]:


#Before turning categorical values to dummy variables. Let's drop columns that we do not need.
train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin','SibSp','Parch', 'Title'], axis=1)
test_df = test_df.drop(['Name', 'Ticket', 'Cabin','SibSp','Parch', 'Title'], axis=1) #We need to keep the PassengerId for these.


# In[33]:


train_df = pd.get_dummies(train_df, drop_first=True)
train_df


# In[34]:


test_df =pd.get_dummies(test_df, drop_first=True)
test_df


# In[35]:


train_df.info()
print("-"*20)
test_df.info()


# In[36]:


#One in test set is missing the Fare. Let's fix it.
test_df.loc[test_df.Fare.isnull(),'Fare']= dataset.Fare.mean()


# In[37]:


test_df.info()


# Now all data cleaning is done and we are ready for making the model for predictions.

# ## Checking different classification models
# To check which model is best we divide the train set to to two pieces. One to building the model and other for testing it. The test_df is the one to which we need to give the predictions and therefore we cannot use that.

# In[38]:


X = train_df.iloc[:,1:].to_numpy()
y = train_df.iloc[:,0].to_numpy()


# In[39]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 1)


# We need to do actually still scaling for the age,fare and familysize so that they are about in same range than the other features.

# In[40]:


from sklearn.preprocessing import StandardScaler   # This is class that automatically does this standardation.
sc = StandardScaler()
X_train[:,1:4] = sc.fit_transform(X_train[:,1:4]) # Here fit calculates the mean and std and transform makes the matrix. 
X_test[:,1:4] = sc.transform(X_test[:,1:4]) # Here we use same mean and std that were used above, because otherwise they would not be comparable.


# ### Logistic regression

# In[41]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# In[42]:


# Checking the accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy: ", round(accuracy_score(y_test, y_pred),4))


# ###K-nearest neigbours

# In[43]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# In[44]:


# Checking the accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy: ", round(accuracy_score(y_test, y_pred),4))


# ### Support vector machine (SVM)

# #### With linear kernel

# In[45]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'linear')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# In[46]:


# Checking the accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy: ", round(accuracy_score(y_test, y_pred),4))


# #### Kernel SVM

# In[47]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# In[48]:


# Checking the accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy: ", round(accuracy_score(y_test, y_pred),4))


# ###Naive Bayes

# In[49]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# In[50]:


# Checking the accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy: ", round(accuracy_score(y_test, y_pred),4))


# ###Decision tree classification

# In[51]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# In[52]:


# Checking the accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy: ", round(accuracy_score(y_test, y_pred),4))


# ###Random forest classification

# In[53]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# In[54]:


# Checking the accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy: ", round(accuracy_score(y_test, y_pred),4))


# ## Choice of the model

# The best model so far is the SVM with non linear rbf-kernel. Let make the final predictions with that.

# In[55]:


#First resetting the X and y so that we can use the whole train_df for training the model.
X = train_df.iloc[:,1:].to_numpy()
y = train_df.iloc[:,0].to_numpy()

#Choosing the correct columns from the official test set.
X_test = test_df.iloc[:,1:].to_numpy()


# In[56]:


# Need to do the standardization again.
sc = StandardScaler()
X[:,1:4] = sc.fit_transform(X[:,1:4]) # Here fit calculates the mean and std and transform makes the matrix. 
X_test[:,1:4] = sc.transform(X_test[:,1:4]) # Here we use same mean and std that were used above, because otherwise they would not be comparable.


# In[57]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
classifier.fit(X,y)


# In[58]:


y_results = classifier.predict(X_test)


# In[59]:


submission_df=pd.concat([test_df['PassengerId'].to_frame() , pd.DataFrame(y_results.reshape((len(y_results),1)), columns=['Survived'])], axis=1)
submission_df.head()


# In[60]:


submission_df.info()


# In[61]:


submission_df['Survived'].value_counts()


# In[62]:


submission_df.to_csv('submission2.csv', index=False)


# This got scored 0.78229. Could use some tuning still.
