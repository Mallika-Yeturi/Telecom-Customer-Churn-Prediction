#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # For creating plots
import matplotlib.ticker as mtick # For specifying the axes tick format 
import matplotlib.pyplot as plt

customer_data = pd.read_csv(r'C:\Users\Mallika Yeturi\OneDrive\Desktop\TelcomCustomer-Churn.csv')
customer_data.head()


# In[2]:


customer_data.info()


# In[4]:


customer_data['SeniorCitizen'] = customer_data['SeniorCitizen'].apply(lambda x: "Yes" if x == 1 else "No")


# In[5]:


customer_data["TotalCharges"] = pd.to_numeric(customer_data["TotalCharges"] , errors = "coerce")
customer_data["TotalCharges"].isnull().sum()


# In[6]:


customer_data.isnull().sum()


# In[7]:


customer_data[np.isnan(customer_data['TotalCharges'])]


# In[8]:


customer_data.drop(labels=customer_data[customer_data['tenure'] == 0].index, axis=0, inplace=True)
customer_data[customer_data['tenure'] == 0].index


# In[9]:


customer_data.fillna(customer_data["TotalCharges"].mean())


# In[10]:


customer_data.isnull().sum()


# In[11]:


customer_data = customer_data.drop(['customerID'], axis = 1)
customer_data.head()


# In[12]:


numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
customer_data[numerical_cols].describe()


# In[13]:


import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.graphics.gofplots import qqplot

def Box_plots(df):
    plt.figure(figsize=(10, 4))
    plt.title(i)
    sns.boxplot(df, orient= "horizontal")
    plt.show()

for i in numerical_cols:
    Box_plots(customer_data[i])


# In[14]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df1 = customer_data.copy(deep = True)
text_data_features = [i for i in customer_data.columns if i not in customer_data.describe().columns]

print('Label Encoder Transformation')
for i in text_data_features :
    df1[i] = le.fit_transform(df1[i])
    print(i,' : ',df1[i].unique(),' = ',le.inverse_transform(df1[i].unique()))


# In[15]:


df_graph = df1.drop("Churn", axis = 1)


# In[16]:


l1 = ['gender','SeniorCitizen','Partner','Dependents'] # Customer Information
l2 = ['PhoneService','MultipleLines','InternetService','StreamingTV','StreamingMovies',
      'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport'] # Services Signed Up for!
l3 = ['Contract','PaperlessBilling','PaymentMethod'] # Payment Information


# In[17]:


import seaborn as sns

# define the datasets
churn = df1[df1['Churn'] == 1].describe().T
not_churn = df1[df1['Churn'] == 0].describe().T
difference = not_churn - churn

#Drop variables
difference = difference.drop('Churn')
churn = churn.drop("Churn")
not_churn = not_churn.drop("Churn")
cmap = sns.color_palette("RdYlGn", as_cmap=True)
#White color palette
white_palette = sns.color_palette("Greys", as_cmap=True)

#Three plots
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 5))
plt.subplot(1, 3, 1)
sns.heatmap(churn[['mean']], annot=True, cmap=white_palette, vmin=0, vmax = 1000000, linewidths=0.4, linecolor='black', cbar=False, fmt='.2f')
plt.title('Churned Customers');

plt.subplot(1, 3, 2)
heatmap = sns.heatmap(difference[['mean']], annot=True, cmap=cmap, vmin=-1, vmax=1, linewidths=0.4, linecolor='black', cbar_kws={"orientation": "vertical", "ticks": [-1, 0, 1]}, fmt='.2f')
plt.title('Difference');
cbar = heatmap.collections[0].colorbar
cbar.set_ticks([-1, 0, 1])
cbar.set_ticklabels(['Churned dominance', 'No difference', 'Not Churned dominance'])

plt.subplot(1, 3, 3)
sns.heatmap(not_churn[['mean']], annot=True, cmap=white_palette, linewidths=0.4, vmin=0, vmax = 1000000, linecolor='black', cbar=False, fmt='.2f',)
plt.title('Not_Churned Customers');

fig.tight_layout(pad=0)


# In[18]:


fig, ax = plt.subplots(figsize=(15, 8))
customer_data['Churn'].replace(to_replace='Yes', value=1, inplace=True)
customer_data['Churn'].replace(to_replace='No',  value=0, inplace=True)
df_dummies = pd.get_dummies(customer_data)

df_dummies_sorted = df_dummies.corr()['Churn'].sort_values(ascending = False)
df_dummies_sorted_desc = df_dummies.corr()['Churn'].sort_values(ascending = True)

df_dummies_sorted.plot(kind='bar', label = True)

df_dummies_strong_above = df_dummies_sorted.apply(lambda x :x if x>0.3 else None).dropna()
strong_above = df_dummies_strong_above.index
df_dummies_strong_above.dropna()

ax.bar(strong_above, df_dummies.corr()['Churn'].loc[strong_above], color='green', alpha=0.5)


# In[19]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(15, 8))

customer_data['Churn'].replace(to_replace='Yes', value=1, inplace=True)
customer_data['Churn'].replace(to_replace='No',  value=0, inplace=True)

df_dummies = pd.get_dummies(customer_data)

df_dummies_sorted = df_dummies.corr()['Churn'].sort_values(ascending=True)

df_dummies_strong_above = df_dummies_sorted.apply(lambda x: x if x > 0.3 else None).dropna()
strong_above = df_dummies_strong_above.index
df_dummies_strong_above.dropna()

df_dummies_strong_below = df_dummies_sorted.apply(lambda x: x if x < -0.3 else None).dropna()
strong_below = df_dummies_strong_below.index
df_dummies_strong_below.dropna()

ax.bar(strong_above, df_dummies.corr()['Churn'].loc[strong_above], color='green', alpha=0.5)
ax.bar(strong_below, df_dummies.corr()['Churn'].loc[strong_below], color='orange', alpha=0.5)

ax.set_xticklabels(df_dummies_sorted.index, rotation=90)
ax.set_ylabel('Correlation with Churn')
ax.set_xlabel('Features')
ax.set_title('Feature Correlation with Churn')

plt.show()


# In[20]:


print(df_dummies_strong_above)


# In[21]:


print(df_dummies_strong_below)


# In[22]:


plt.figure(figsize=(25, 10))

corr = df1.apply(lambda x: pd.factorize(x)[0]).corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

ax = sns.heatmap(corr, mask=mask, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, linewidths=.2, cmap='coolwarm', vmin=-1, vmax=1)


# In[23]:


fig, ax = plt.subplots(nrows = 1,ncols = 3,figsize = (25,5))

colors = ['#E94B3C','#2D2926']

for i in range(len(numerical_cols)):
    plt.subplot(1,3,i+1)
    sns.distplot(df1[numerical_cols[i]],color = colors[0])
    title = 'Distribution : ' + numerical_cols[i]
    plt.title(title)
plt.show()


# In[24]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(20, 8))
sns.countplot(x="tenure", hue="Churn", data=df1)
plt.legend(["No Churn"], loc = "upper left")
plt.show()


# In[25]:


df1['MonthlyCharges_Group'] = [int(i / 5) for i in df1['MonthlyCharges']]
df1['TotalCharges_Group'] = [int(i / 500) for i in df1['TotalCharges']]


sns.countplot(x="TotalCharges_Group", hue = "Churn", data=df1, palette = colors,edgecolor = 'black')
plt.legend(['No Churn','Churn'],loc = 'upper left')
title = numerical_features[1 + i] + ' w.r.t Churn'
plt.title(title);


# In[26]:


df1['MonthlyCharges_Group'] = [int(i / 5) for i in df1['MonthlyCharges']]
df1['TotalCharges_Group'] = [int(i / 500) for i in df1['TotalCharges']]


sns.countplot(x="MonthlyCharges_Group", hue = "Churn", data=df1, palette = colors,edgecolor = 'black')
plt.legend(['No Churn','Churn'],loc = 'upper left')
title = numerical_features[1 + i] + ' w.r.t Churn'
plt.title(title);


# In[61]:


sns.set_context("paper",font_scale=1.1)
ax = sns.kdeplot(df1.MonthlyCharges[(df1["Churn"] == 0) ],
                color="Red", shade = True);
ax = sns.kdeplot(df1.MonthlyCharges[(df1["Churn"] == 1) ],
                ax =ax, color="Blue", shade= True);

ax.axhline(y=0.01, color="black", linestyle="--")

ax.legend(["Not Churn","Churn"],loc='upper right');
ax.set_ylabel('Density');
ax.set_xlabel('Monthly Charges');
ax.set_title('Distribution of monthly charges by churn');


# In[29]:


ax = sns.kdeplot(df1.TotalCharges[(df1["Churn"] == 0) ],
                color="Gold", shade = True);
ax = sns.kdeplot(df1.TotalCharges[(df1["Churn"] == 1) ],
                ax =ax, color="Green", shade= True);
ax.legend(["Not Chu0rn","Churn"],loc='upper right');
ax.set_ylabel('Density');
ax.set_xlabel('Total Charges');
ax.set_title('Distribution of total charges by churn');


# In[30]:


plt.figure(figsize=(6, 6))
labels =["Churn: Yes","Churn:No"]
values = [1869,5163]
labels_gender = ["F","M","F","M"]
sizes_gender = [939,930 , 2544,2619]
colors = ['#ff6666', '#66b3ff']
colors_gender = ['#c2c2f0','#ffb3e6', '#c2c2f0','#ffb3e6']
explode = (0.3,0.3) 
explode_gender = (0.1,0.1,0.1,0.1)
textprops = {"fontsize":15}
#Plot
plt.pie(values, labels=labels,autopct='%1.1f%%',pctdistance=1.08, labeldistance=0.8,colors=colors, startangle=90,frame=True, explode=explode,radius=10, textprops =textprops, counterclock = True, )
plt.pie(sizes_gender,labels=labels_gender,colors=colors_gender,startangle=90, explode=explode_gender,radius=7, textprops =textprops, counterclock = True, )
#Draw circle
centre_circle = plt.Circle((0,0),5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title('Churn Distribution w.r.t Gender: Male(M), Female(F)', fontsize=15, y=1.1)

# show plot 
 
plt.axis('equal')
plt.tight_layout()
plt.show()


# In[31]:


fig = plt.subplots(nrows = 2,ncols = 2,figsize = (15,10))
for i in range(4):
    plt.subplot(2,2,i+1)
    ax = sns.boxplot(x = l1[i],y = 'tenure',data = customer_data,hue = 'Churn',palette = colors);
    plt.title('tenure vs ' + l1[i]);


# In[32]:


fig = plt.subplots(nrows = 1,ncols = 2,figsize = (15,5))

for i in range(len(l2[0:2])):
    plt.subplot(1,2,i + 1)
    ax = sns.boxplot(x = l2[i],y = 'tenure',data = customer_data,hue = 'Churn',palette = colors);
    plt.title('tenure vs ' + l2[i]);

fig = plt.subplots(nrows = 1, ncols = 1, figsize = (6,5))

plt.subplot(1,1,1)
ax = sns.boxplot(x = l2[2],y = 'tenure',data = customer_data,hue = 'Churn',palette = colors);
plt.title('tenure vs ' + l2[2]);
    
fig = plt.subplots(nrows = 1,ncols = 2,figsize = (12,5))

for i in range(len(l2[3:5])):
    plt.subplot(1,2,i + 1)
    ax = sns.boxplot(x = l2[i + 3],y = 'tenure',data = customer_data,hue = 'Churn',palette = colors);
    plt.title('tenure vs ' + l2[i + 3]);


# In[33]:


fig = plt.subplots(nrows = 2,ncols = 2,figsize = (20,14))
for i in range(len(l2[-4:])):
    plt.subplot(2,2,i + 1)
    ax = sns.boxplot(x = l2[i - 4],y = 'tenure',data = customer_data,hue = 'Churn',palette = colors);
    plt.title('tenure vs ' + l2[i-4]);


# In[34]:


fig = plt.subplots(nrows = 1,ncols = 3,figsize = (30,7))
for i in range(len(l3)):
    plt.subplot(1,3,i + 1)
    ax = sns.boxplot(x = l3[i],y = 'tenure',data = customer_data,hue = 'Churn',palette = colors);
    plt.title('tenure vs ' + l3[i]);


# In[35]:


fig = plt.subplots(nrows = 1,ncols = 2,figsize = (15,5))

for i in range(len(l2[0:2])):
    plt.subplot(1,2,i + 1)
    ax = sns.boxplot(x = l2[i],y = 'MonthlyCharges',data = customer_data,hue = 'Churn',palette = colors);
    plt.title('MonthlyCharges vs ' + l2[i]);

fig = plt.subplots(nrows = 1, ncols = 1, figsize = (6,5))

plt.subplot(1,1,1)
ax = sns.boxplot(x = l2[2],y = 'MonthlyCharges',data = customer_data,hue = 'Churn',palette = colors);
plt.title('MonthlyCharges vs ' + l2[2]);
    
fig = plt.subplots(nrows = 1,ncols = 2,figsize = (12,5))

for i in range(len(l2[3:5])):
    plt.subplot(1,2,i + 1)
    ax = sns.boxplot(x = l2[i + 3],y = 'MonthlyCharges',data = customer_data,hue = 'Churn',palette = colors);
    plt.title('MonthlyCharges vs ' + l2[i + 3]);


# In[36]:


fig = plt.subplots(nrows = 2,ncols = 2,figsize = (20,14))
for i in range(len(l2[-4:])):
    plt.subplot(2,2,i + 1)
    ax = sns.boxplot(x = l2[i - 4],y = 'MonthlyCharges',data = customer_data,hue = 'Churn',palette = colors);
    plt.title('MonthlyCharges vs ' + l2[i-4]);


# In[37]:


fig = plt.subplots(nrows = 1,ncols = 3,figsize = (25,7))

for i in range(len(l3)):
    plt.subplot(1,3,i + 1)
    ax = sns.boxplot(x = l3[i],y = 'MonthlyCharges',data = customer_data,hue = 'Churn',palette = colors);
    title = 'MonthlyCharges vs ' + l3[i]
    plt.title(title);


# In[38]:


fig = plt.subplots(nrows = 1,ncols = 2,figsize = (15,5))

for i in range(len(l2[0:2])):
    plt.subplot(1,2,i + 1)
    ax = sns.boxplot(x = l2[i],y = 'TotalCharges',data = customer_data,hue = 'Churn',palette = colors);
    plt.title('TotalCharges vs ' + l2[i]);

fig = plt.subplots(nrows = 1, ncols = 1, figsize = (6,5))

plt.subplot(1,1,1)
ax = sns.boxplot(x = l2[2],y = 'TotalCharges',data = customer_data,hue = 'Churn',palette = colors);
plt.title('TotalCharges vs ' + l2[2]);
    
fig = plt.subplots(nrows = 1,ncols = 2,figsize = (12,5))

for i in range(len(l2[3:5])):
    plt.subplot(1,2,i + 1)
    ax = sns.boxplot(x = l2[i + 3],y = 'TotalCharges',data = customer_data,hue = 'Churn',palette = colors);
    plt.title('TotalCharges vs ' + l2[i + 3]);


# In[39]:


fig = plt.subplots(nrows = 2,ncols = 2,figsize = (20,14))
for i in range(len(l2[-4:])):
    plt.subplot(2,2,i + 1)
    ax = sns.boxplot(x = l2[i - 4],y = 'TotalCharges',data = customer_data,hue = 'Churn',palette = colors);
    plt.title('TotalCharges vs ' + l2[i-4]);


# In[40]:


fig = plt.subplots(nrows = 1,ncols = 3,figsize = (30,7))
for i in range(len(l3)):
    plt.subplot(1,3,i + 1)
    ax = sns.boxplot(x = l3[i],y = 'TotalCharges',data = customer_data,hue = 'Churn',palette = colors);
    plt.title('TotalCharges vs ' + l3[i]);


# In[42]:


from sklearn.preprocessing import MinMaxScaler,StandardScaler
mms = MinMaxScaler() # Normalization
ss = StandardScaler() # Standardization

df1.drop(columns = ['MonthlyCharges_Group','TotalCharges_Group'], inplace = True)

df1['tenure'] = mms.fit_transform(df1[['tenure']])
df1['MonthlyCharges'] = mms.fit_transform(df1[['MonthlyCharges']])
df1['TotalCharges'] = mms.fit_transform(df1[['TotalCharges']])
df1.head()


# In[43]:


plt.figure(figsize = (20,5))
sns.heatmap(df1.corr(),cmap = colors,annot = True);


# In[44]:


corr = df1.corrwith(df1['Churn']).sort_values(ascending = False).to_frame()
corr.columns = ['Correlations']
plt.subplots(figsize = (5,5))
sns.heatmap(corr,annot = True,cmap = colors,linewidths = 0.4,linecolor = 'black');
plt.title('Correlation w.r.t Outcome');


# In[45]:


from scipy import stats
alpha = 0.05
features = ["tenure", "TotalCharges", "MonthlyCharges", "Contract", "PaperlessBilling"]


null_hypothesis = "The users' characteristics as well as the type of their plan bear no correlation to their churning rate"
alternative_hypothesis = "certain variables such as tenure, total charge, monthly charge as well as package variables affect the churning rate and can be used to predict whether the user will churn."

for i in features:
    t_statistic, p_value = stats.ttest_ind(customer_data["tenure"], customer_data["Churn"])
    print(t_statistic, p_value)

    if p_value <= alpha:
        print("Reject the null hypothesis: ", alternative_hypothesis)
    else:
        print("Fail to reject the null hypothesis: ", null_hypothesis)


# In[46]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,mutual_info_classif


from sklearn.feature_selection import f_classif


# In[51]:


df1.drop(columns = ['PhoneService', 'gender','StreamingTV','StreamingMovies','MultipleLines','InternetService'],inplace = True)
df1.head()


# In[52]:


import imblearn
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


# In[53]:


over = SMOTE(sampling_strategy = 1)

#Everything else
f1 = df1.iloc[:,:13].values
#Churn
t1 = df1.iloc[:,13].values

f1, t1 = over.fit_resample(f1, t1)
Counter(t1)


# In[55]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_recall_curve


# In[56]:


#Spliting code into train/test split
x_train, x_test, y_train, y_test = train_test_split(f1, t1, test_size = 0.20, random_state = 2)


# In[57]:


# Cross-validation + ROC_AUC score
def model(classifier,x_train,y_train,x_test,y_test):
    
    classifier.fit(x_train,y_train)
    prediction = classifier.predict(x_test)
    cv = RepeatedStratifiedKFold(n_splits = 10,n_repeats = 3,random_state = 1)
    print("Cross Validation Score : ",'{0:.2%}'.format(cross_val_score(classifier,x_train,y_train,cv = cv,scoring = 'roc_auc').mean()))
    print("ROC_AUC Score : ",'{0:.2%}'.format(roc_auc_score(y_test,prediction)))
    plot_roc_curve(classifier, x_test,y_test)
    plt.title('ROC_AUC_Plot')
    plt.show()

def model_evaluation(classifier,x_test,y_test):
    
    # Confusion Matrix
    cm = confusion_matrix(y_test,classifier.predict(x_test))
    names = ['True Neg','False Pos','False Neg','True Pos']
    counts = [value for value in cm.flatten()]
    percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cm,annot = labels,cmap = 'Blues',fmt ='')
    
    # Classification Report
    print(classification_report(y_test,classifier.predict(x_test)))


# In[64]:


ax = sns.kdeplot(customer_data.TotalCharges[(customer_data["Churn"] == 'No') ],
                color="Red", shade = True)
ax = sns.kdeplot(customer_data.TotalCharges[(customer_data["Churn"] == 'Yes') ],
                ax =ax, color="Blue", shade= True)
ax.legend(["Not Churn","Churn"],loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('Total Charges')
ax.set_title('Distribution of total charges by churn')


# In[65]:


y = df_dummies['Churn'].values
X = df_dummies.drop(columns = ['Churn'])

# Scaling all the variables to a range of 0 to 1
from sklearn.preprocessing import MinMaxScaler
features = X.columns.values
scaler = MinMaxScaler(feature_range = (0,1))
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X))
X.columns = features


# In[66]:


# Create Train & Test Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[67]:


# Running logistic regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
result = model.fit(X_train, y_train)


# In[68]:


from sklearn import metrics
prediction_test = model.predict(X_test)
# Print the prediction accuracy
print (metrics.accuracy_score(y_test, prediction_test))


# In[69]:


# To get the weights of all the variables
weights = pd.Series(model.coef_[0],
                 index=X.columns.values)
print (weights.sort_values(ascending = False)[:10].plot(kind='bar'))


# In[70]:


print(weights.sort_values(ascending = False)[-10:].plot(kind='bar'))


# In[71]:


from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
model_rf = RandomForestClassifier(n_estimators=1000 , oob_score = True, n_jobs = -1,
                                  random_state =50, max_features = "auto",
                                  max_leaf_nodes = 30)
model_rf.fit(X_train, y_train)

# Make predictions
prediction_test = model_rf.predict(X_test)
print (metrics.accuracy_score(y_test, prediction_test))


# In[72]:


importances = model_rf.feature_importances_
weights = pd.Series(importances,
                 index=X.columns.values)
weights.sort_values()[-10:].plot(kind = 'barh')


# In[73]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)


# In[74]:


from sklearn.svm import SVC

model.svm = SVC(kernel='linear') 
model.svm.fit(X_train,y_train)
preds = model.svm.predict(X_test)
metrics.accuracy_score(y_test, preds)


# In[75]:


from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,preds)) 


# In[77]:


ax1 = sns.catplot(x="gender", kind="count", hue="Churn", data=customer_data,
                  estimator=lambda x: sum(x==0)*100.0/len(x))
#ax1.yaxis.set_major_formatter(mtick.PercentFormatter())


# In[78]:


# AdaBoost Algorithm
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
# n_estimators = 50 (default value) 
# base_estimator = DecisionTreeClassifier (default value)
model.fit(X_train,y_train)
preds = model.predict(X_test)
metrics.accuracy_score(y_test, preds)

# Telco should focus on:
1. The 70/Month Price Mark is the largest indicator for churning. The customer retention team should focus on keeping users who want to switch to a different provider below this number and offer a competetive rate
2. OnlineSecurity, OnlineBackup, DeviceProtection & TechSupport as having these packages can be a deciding factor for the first 10 months. And focus on the 40-50 month time period as that is when churning for those who are opted-in happens
3. Focus on Multiple lines & Fiber Optic as those options correspond with users staying longer. (Perhaps the high price of Fiber Optic scares off some users, so that needs further investigating)
4. Focus on Bank Transfer/Credit Card as those are less expensive than elctronic check
5. A package system of multiple lines/Fiber Optic to stay below the 100/110 range as that will avoid churning due to high monthly bills.