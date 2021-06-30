import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as rd
import statsmodels.api as sm
import scipy.stats as stats
#%% Importing Data
# Raw Data (Would not be modified)
TrainRawData = pd.read_csv(r'train.csv', header=0)
TestRawData = pd.read_csv(r'test.csv', header=0)

# Modified Data
TrainData = pd.read_csv(r'train.csv', header=0)
TestData = pd.read_csv(r'test.csv', header=0)
#%% Scatter Matrix
pd.plotting.scatter_matrix(TrainRawData, alpha=0.2)
#%% Relation between variables
train_pclass = TrainRawData['Pclass']
train_sex = TrainRawData['Sex']
train_age = TrainRawData['Age']
train_Sib = TrainRawData['SibSp']
#%% Sex and Survived
# Data Engineering
df_male = TrainRawData.loc[TrainRawData['Sex'] == 'male']
df_female = TrainRawData.loc[TrainRawData['Sex'] == 'female']

df_male_sur = df_male['Survived']
df_female_sur = df_female['Survived']

count_male_nSur = len(df_male.loc[df_male['Survived'] == 0])
count_male_Sur = len(df_male.loc[df_male['Survived'] == 1])

count_female_nSur = len(df_female.loc[df_female['Survived'] == 0])
count_female_Sur = len(df_female.loc[df_female['Survived'] == 1])

sexs = ['Male','Female']
SurValue = [count_male_Sur,count_female_Sur]
nSurValue = [count_male_nSur,count_female_nSur]

# Plotting
x = np.arange(len(sexs))
width = 0.3

plt.bar(x,SurValue,width=width,color='red',label='Survived',alpha=0.5)
plt.bar(x+width,nSurValue,width=width,color='blue',label='Not Survived',alpha=0.5)
plt.xticks(x+width/2,sexs)
plt.ylabel('Count')
plt.title('Barplot of Sex Gruoped by Survived')
plt.legend()
plt.show()
#%% Sex Converting to Value
temp = []
for i in range(len(TrainData)):
    if TrainData['Sex'].iloc[i] == 'male':
        temp.append(0)
    else:
        temp.append(1)
TrainData['SexValue'] = temp

temp2 = []
for i in range(len(TestData)):
    if TestData['Sex'].iloc[i] == 'male':
        temp2.append(0)
    else:
        temp2.append(1)
TestData['SexValue'] = temp2
#%% Age and Survived (Survived=274,numNA=46 ; not Survived=428,numNA=143)
# Data Engineering
df_Sur = TrainRawData.loc[TrainRawData['Survived'] == 1]
df_nSur = TrainRawData.loc[TrainRawData['Survived'] == 0]

df_Age_nNA = TrainRawData['Age'].dropna()
df_SurAge = df_Sur['Age']
df_SurAge_nNA = df_SurAge.dropna()
df_nSurAge = df_nSur['Age']
df_nSurAge_nNA = df_nSurAge.dropna()

# Plotting
plt.hist(df_Age_nNA,alpha=0.3,label='Total',bins=30,density=False)
plt.hist(df_SurAge_nNA,alpha=0.5,label='Survived',bins=30,density=False)
plt.hist(df_nSurAge_nNA,alpha=0.5,label='Not Survived',bins=30,density=False)
plt.title('Multiple Histrogram of Age Grouped by Total, Survived, and Not Survived')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(loc='best')
plt.show()
#%% Total Survived Rate
# Plotting and Calculating
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i])
TotSurRate = len(df_Sur)/len(TrainData)
Survived = ['Survived','Not Survived']
x = np.arange(len(Survived))
CountSurRate = [len(df_Sur),len(df_nSur)]
plt.bar(Survived,CountSurRate,color='red',alpha=0.5)
addlabels(Survived, CountSurRate)
plt.title('Survived and Not Survived Barplot in Total Population')
plt.ylabel('Count')
plt.show()
print('Total Survived Rate:',TotSurRate)
#%% Survival Rate in Specific Age interval
# Plotting and Calculating
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i])
def CountSur(min_in,max_in):
    df_int_age_pre = TrainData.loc[TrainData['Age'] <= max_in]
    df_int_age = df_int_age_pre.loc[TrainData['Age'] >= min_in]
    df_int_age_Sur = df_int_age.loc[df_int_age['Survived'] == 1]
    df_int_age_nSur = df_int_age.loc[df_int_age['Survived'] == 0]
    count_int_age_Sur = len(df_int_age_Sur)
    count_int_age_nSur = len(df_int_age_nSur)
    int_SurRate = count_int_age_Sur/(count_int_age_Sur+count_int_age_nSur)
    Survived = ['Survived','Not Survived']
    count_int_age = [count_int_age_Sur,count_int_age_nSur]
    plt.bar(Survived,count_int_age,color='blue',alpha=0.5)
    addlabels(Survived, count_int_age)
    plt.title('Survived and Not Survived Barplot')
    plt.ylabel('Count')
    plt.show()
    print('Survival Rate between',min_in,'and',max_in,':',int_SurRate)
    return int_SurRate
# CountSur(0,15)
temp = []
for i in range(10):
    #CountSur(16+5*i,20+5*i)
    temp.append(CountSur(16+5*i,20+5*i))
CountSur(66,100)
print(temp)
#%% Age NA situation
#Data Engineering
df_age_temp = df_Age_nNA.sample(n=len(TrainRawData)-len(df_Age_nNA))
df_age_NA = TrainRawData.drop(index=df_Age_nNA.index)
num_NA = len(TrainRawData)-len(df_Age_nNA)
por_NA = int(num_NA/3)
print('Number of NA in Age column:',num_NA)
print('Part of NA in Age column:',por_NA)
index_age_NA = df_age_NA.index

dfTest_Age_nNA = TestRawData['Age'].dropna()
dfTest_age_NA = TestRawData.drop(index=dfTest_Age_nNA.index)
indexTest_age_NA = dfTest_age_NA.index

num_NATest = len(TestRawData)-len(dfTest_Age_nNA)
por_NATest = int(num_NATest/3)
print('Number of NA in Age column:',num_NATest)
print('Part of NA in Age column:',por_NATest)
index_age_NA = df_age_NA.index
#print(index_age_NA[1])
#%% Age NA modified by the random numbers 3 specific intervals
temp_age = []
temp_age_1 = rd.randint(25,31,size=por_NA)
temp_age_2 = rd.randint(31,36,size=por_NA)
temp_age_3 = rd.randint(51,56,size=por_NA)
temp_age.append(temp_age_1)
temp_age.append(temp_age_2)
temp_age.append(temp_age_3)
temp_age = np.reshape(temp_age,num_NA,order='F')
rd.shuffle(temp_age)
#print(temp_age[2])
for i in range(num_NA):
    TrainData['Age'].loc[index_age_NA[i]] = temp_age[i]

temp_ageTest = []
temp_age_1Test = rd.randint(21,26,size=por_NATest+1)
temp_age_2Test = rd.randint(31,36,size=por_NATest+1)
temp_age_3Test = rd.randint(46,51,size=por_NATest+1)
temp_ageTest.append(temp_age_1Test)
temp_ageTest.append(temp_age_2Test)
temp_ageTest.append(temp_age_3Test)
temp_ageTest = np.reshape(temp_ageTest,75,order='F')
rd.shuffle(temp_ageTest)
#print(temp_age[2])
for i in range(num_NATest):
    TestData['Age'].loc[indexTest_age_NA[i]] = temp_ageTest[i]
#%% AgeLabel for Random Forest
temp_AgeLabel = []
for i in range(len(TrainData)):
    if TrainData['Age'].iloc[i] < 16:
        temp_AgeLabel.append(0)
    elif TrainData['Age'].iloc[i] > 65:
        temp_AgeLabel.append(11)
    else:
        for j in range(1,11):
            if TrainData['Age'].iloc[i] >= (11+5*j):
                if TrainData['Age'].iloc[i] < (16+5*j):
                    temp_AgeLabel.append(j)
                    break
TrainData['AgeLabel'] = temp_AgeLabel

temp_AgeLabelTest = []
for i in range(len(TestData)):
    if TestData['Age'].iloc[i] < 16:
        temp_AgeLabelTest.append(0)
    elif TestData['Age'].iloc[i] > 65:
        temp_AgeLabelTest.append(11)
    else:
        for j in range(1,11):
            if TestData['Age'].iloc[i] >= (11+5*j):
                if TestData['Age'].iloc[i] < (16+5*j):
                    temp_AgeLabelTest.append(j)
                    break
TestData['AgeLabel'] = temp_AgeLabelTest
#%% Age Missing Value Modified by Bootstrap Method (Not helpful for predicting model)
#dfTest_Age_nNA = TestRawData['Age'].dropna()
#dfTest_age_temp = df_Age_nNA.sample(n=len(TestRawData)-len(dfTest_Age_nNA))
#dfTest_age_NA = TestRawData.drop(index=dfTest_Age_nNA.index)
#indexTest_age_NA = dfTest_age_NA.index

#for i in range(len(df_age_NA)):
#    TrainData['Age'].loc[index_age_NA[i]] = df_age_temp.iloc[i]

#for i in range(len(df_age_NA)):
#    TestData['Age'].loc[indexTest_age_NA[i]] = dfTest_age_temp.iloc[i]
#%% New "Title" Column
TrainData['Title'] = TrainData.Name.str.split(', ', expand=True)[1]
TrainData['Title'] = TrainData.Title.str.split('.', expand=True)[0]
TrainData['Title'].unique()

TestData['Title'] = TestData.Name.str.split(', ', expand=True)[1]
TestData['Title'] = TestData.Title.str.split('.', expand=True)[0]
TestData['Title'].unique()
#%% Median and Mean Age of Each Title
print(TrainData['Title'].describe())
arr_title = TrainData['Title'].unique()
print(len(arr_title))
arr_title_median = []
arr_title_mean = []
for i in range(len(arr_title)):
    temp_title = []
    for j in range(len(TrainData)):
        if TrainData['Title'].iloc[j] == arr_title[i]:
            temp_title.append(TrainData['Age'].iloc[j])
    arr_title_median.append(int(np.median(temp_title)))
    arr_title_mean.append(int(np.mean(temp_title)))
print(arr_title)
print(arr_title_median)
print(arr_title_mean)
# for i in range(len(df_age_NA)):
# TrainData['Age'].loc[index_age_NA[i]] = df_age_temp.iloc[i]
#%% Modifying Age by Title
TrainData['Age_byMedian'] = TrainRawData['Age']
TrainData['Age_byMean'] = TrainRawData['Age']
TestData['Age_byMedian'] = TestRawData['Age']
TestData['Age_byMean'] = TestRawData['Age']


for i in range(len(index_age_NA)):
    for j in range(len(arr_title)):
        if TrainData['Title'].iloc[index_age_NA[i]] == arr_title[j]:
            TrainData['Age_byMedian'].iloc[index_age_NA[i]] = arr_title_median[j]
            TrainData['Age_byMean'].iloc[index_age_NA[i]] = arr_title_mean[j]
            
for i in range(len(indexTest_age_NA)):
    for j in range(len(arr_title)):
        if TestData['Title'].iloc[indexTest_age_NA[i]] == arr_title[j]:
            TestData['Age_byMedian'].iloc[indexTest_age_NA[i]] = arr_title_median[j]
            TestData['Age_byMean'].iloc[indexTest_age_NA[i]] = arr_title_mean[j]
#%% Plotting with two histogram
df_age = TrainData['Age']

plt.grid()
plt.hist(df_Age_nNA,alpha=0.3,label='Valid(Raw Data w/o NA)',bins=30,density=True,color='Red')
plt.hist(df_age,alpha=0.5,label='Modified',bins=30,density=True)
plt.xlabel('Age')
plt.ylabel('Density')
plt.title('Density Distribution of Original Valid Data and Modified Data')
plt.legend(loc='best')
plt.show()
#%% Age Labeling for Random Forest by two gruops
temp = []
for i in range(len(TrainData)):
    if TrainData['Age'].iloc[i] <= 16:
        temp.append('Young')
    else:
        temp.append('Older')
TrainData['AgeLabel2'] = temp

temp2 = []
for i in range(len(TestData)):
    if TestData['Age'].iloc[i] <= 16:
        temp2.append('Young')
    else:
        temp2.append('Older')
TestData['AgeLabel2'] = temp2
#%% Make the Quantile-Quantile(QQ) plot to check normality of Age
#import statsmodels.api as sm
#import scipy.stats as stats
sm.qqplot(TrainData['Age'],stats.t,fit=True, line='45')
plt.title('QQ-plot of Age')
plt.show()
#%% Pclass Describe and Plotting
df_P1 = TrainRawData.loc[TrainRawData['Pclass'] == 1]
df_P2 = TrainRawData.loc[TrainRawData['Pclass'] == 2]
df_P3 = TrainRawData.loc[TrainRawData['Pclass'] == 3]

dfTest_P1 = TestRawData.loc[TrainRawData['Pclass'] == 1]
dfTest_P2 = TestRawData.loc[TrainRawData['Pclass'] == 2]
dfTest_P3 = TestRawData.loc[TrainRawData['Pclass'] == 3]

count_P1_nSur = len(df_P1.loc[df_P1['Survived'] == 0])
count_P1_Sur = len(df_P1.loc[df_P1['Survived'] == 1])
count_P2_nSur = len(df_P2.loc[df_P2['Survived'] == 0])
count_P2_Sur = len(df_P2.loc[df_P2['Survived'] == 1])
count_P3_nSur = len(df_P3.loc[df_P3['Survived'] == 0])
count_P3_Sur = len(df_P3.loc[df_P3['Survived'] == 1])

SurValue = [count_P1_Sur,count_P2_Sur,count_P3_Sur]
nSurValue = [count_P1_nSur,count_P2_nSur,count_P3_nSur]

Pclass = ['Pclass 1','Pclass 2','Pclass 3']

x = np.arange(len(Pclass))
width = 0.4

plt.bar(x,SurValue,width=width,color='red',label='Survived',alpha=0.5)
plt.bar(x+width,nSurValue,width=width,color='blue',label='Not Survived',alpha=0.5)
plt.xticks(x+width/3,Pclass)
plt.ylabel('Count')
plt.title('Pclass Barplot Gruoped by Survived in Original Data')
plt.legend()
plt.show()
#%% Pclass Missing Value Modified by Proportion (Not Adaptive)
df_Pclass_nNA = TrainRawData['Pclass'].dropna()
df_Pclass_temp = df_Pclass_nNA.sample(n=len(TrainRawData)-len(df_Pclass_nNA))
df_Pclass_NA = TrainRawData.drop(index=df_Pclass_nNA.index)

index_Pclass_NA = df_Pclass_NA.index

dfTest_Pclass_nNA = TestRawData['Pclass'].dropna()
dfTest_Pclass_NA = TestRawData.drop(index=dfTest_Pclass_nNA.index)
indexTest_Pclass_NA = dfTest_Pclass_NA.index
# print(index_Pclass_NA[1])
for i in range(len(df_Pclass_NA)):
    TrainData['Pclass'].loc[index_Pclass_NA[i]] = df_Pclass_temp.iloc[i]
#%% Fare logarithm
df_Fare_nNA = TrainRawData['Fare'].dropna()
df_Fare_NA = TrainRawData.drop(index=df_Fare_nNA.index)
print(df_Fare_NA.index)
TrainData['Fare'].loc[df_Fare_NA.index] = df_Fare_nNA.mean()

temp = []
for i in range(len(TrainData)):
    if TrainData['Fare'].iloc[i] != 0:
        logfare = np.log(TrainData['Fare'].iloc[i])
        temp.append(logfare)
    else:
        temp.append(0)
TrainData['LogFare'] = temp

temp2 = []
for i in range(len(TestData)):
    if TestData['Fare'].iloc[i] != 0:
        logfare = np.log(TestData['Fare'].iloc[i])
        temp2.append(logfare)
    else:
        temp2.append(0)
TestData['LogFare'] = temp2
#%% LogFare boxplot of three pclass
df_p1_LogFare = TrainData['LogFare'].loc[df_P1.index]
df_p2_LogFare = TrainData['LogFare'].loc[df_P2.index]
df_p3_LogFare = TrainData['LogFare'].loc[df_P3.index]

data = [df_p1_LogFare,df_p2_LogFare,df_p3_LogFare]
#print(data)

x_1 = np.arange(1,len(Pclass)+1)
plt.boxplot(data)
plt.title('LogFare Boxplot of 3 Plcass')
plt.xticks(x_1,Pclass)
plt.ylabel('log(Fare)')
plt.show()

df_p1_Fare = TrainData['Fare'].loc[df_P1.index]
df_p2_Fare = TrainData['Fare'].loc[df_P2.index]
df_p3_Fare = TrainData['Fare'].loc[df_P3.index]

data1 = [df_p1_Fare,df_p2_Fare,df_p3_Fare]

x_1 = np.arange(1,len(Pclass)+1)
plt.boxplot(data1)
plt.title('Fare Boxplot of 3 Plcass')
plt.xticks(x_1,Pclass)
plt.ylabel('Fare')
plt.show()

df_p2_LogFare_sorted = df_p2_LogFare.sort_values()
print(df_p2_LogFare_sorted.iloc[3])
#%% Normality test, ANOVA test for LogFare of three Pclass
# import scipy.stats as stats
print(stats.shapiro(df_p1_LogFare))
print(stats.shapiro(df_p2_LogFare))
print(stats.shapiro(df_p3_LogFare))

f_value, p_value = stats.f_oneway(df_p1_LogFare, df_p2_LogFare, df_p3_LogFare)
print(p_value)
# t-Test for each two kinds of LogFare grouped by Pclass
print(stats.ttest_ind(df_p1_LogFare, df_p2_LogFare))
print(stats.ttest_ind(df_p2_LogFare, df_p3_LogFare))
print(stats.ttest_ind(df_p1_LogFare, df_p3_LogFare))
#%% LogFare boxplot of three pclass by Survived Group
df_p1_LogFare_Sur = df_p1_LogFare.loc[df_P1['Survived']==1]
df_p1_LogFare_nSur = df_p1_LogFare.loc[df_P1['Survived']==0]
df_p2_LogFare_Sur = df_p2_LogFare.loc[df_P2['Survived']==1]
df_p2_LogFare_nSur = df_p2_LogFare.loc[df_P2['Survived']==0]
df_p3_LogFare_Sur = df_p3_LogFare.loc[df_P3['Survived']==1]
df_p3_LogFare_nSur = df_p3_LogFare.loc[df_P3['Survived']==0]

data_Sur = [df_p1_LogFare_Sur,df_p2_LogFare_Sur,df_p3_LogFare_Sur]
data_nSur = [df_p1_LogFare_nSur,df_p2_LogFare_nSur,df_p3_LogFare_nSur]

pos_Sur = np.arange(0.5,2*len(Pclass),2)
pos_nSur = np.arange(1.3,2*len(Pclass)+1.3,2) 

plt.boxplot(data_Sur,positions=pos_Sur)
plt.boxplot(data_nSur,positions=pos_nSur)
plt.title('LogFare Boxplot of 3 Plcass by Survived Group (L-1,R-0)')
x_2 = np.arange(-1,2*len(Pclass),2)
Pclass_2 = [' ','Pclass 1','Pclass 2','Pclass 3']
plt.xticks(x_2,Pclass_2)
plt.ylabel('log(Fare)')
plt.show()
#%% Missing Pclass Modified by LogFare
for i in range(len(df_Pclass_NA)):
    if (TrainData['LogFare'].loc[index_Pclass_NA[i]] <
        df_p2_LogFare_sorted.iloc[3]):
        TrainData['Pclass'].loc[index_Pclass_NA[i]] = 1
    elif (df_p2_LogFare_sorted.iloc[3] <= 
          TrainData['LogFare'].loc[index_Pclass_NA[i]] <= 
          df_p2_LogFare_sorted.iloc[len(df_Pclass_NA)]):
        TrainData['Pclass'].loc[index_Pclass_NA[i]] = 2
    else:
        TrainData['Pclass'].loc[index_Pclass_NA[i]] = 3
        
for i in range(len(dfTest_Pclass_NA)):
    if (TestData['LogFare'].loc[indexTest_Pclass_NA[i]] <
        df_p2_LogFare_sorted.iloc[3]):
        TestData['Pclass'].loc[indexTest_Pclass_NA[i]] = 1
    elif (df_p2_LogFare_sorted.iloc[3] <= 
          TestData['LogFare'].loc[indexTest_Pclass_NA[i]] <= 
          df_p2_LogFare_sorted.iloc[len(df_Pclass_NA)]):
        TestData['Pclass'].loc[indexTest_Pclass_NA[i]] = 2
    else:
        TestData['Pclass'].loc[indexTest_Pclass_NA[i]] = 3
#%% Plotting LogFare boxplot of three pclass by Survived Group Again
df_P1_m = TrainData.loc[TrainData['Pclass'] == 1]
df_P2_m = TrainData.loc[TrainData['Pclass'] == 2]
df_P3_m = TrainData.loc[TrainData['Pclass'] == 3]

count_P1_m_nSur = len(df_P1_m.loc[df_P1_m['Survived'] == 0])
count_P1_m_Sur = len(df_P1_m.loc[df_P1_m['Survived'] == 1])
count_P2_m_nSur = len(df_P2_m.loc[df_P2_m['Survived'] == 0])
count_P2_m_Sur = len(df_P2_m.loc[df_P2_m['Survived'] == 1])
count_P3_m_nSur = len(df_P3_m.loc[df_P3_m['Survived'] == 0])
count_P3_m_Sur = len(df_P3_m.loc[df_P3_m['Survived'] == 1])

SurValue_m = [count_P1_m_Sur,count_P2_m_Sur,count_P3_m_Sur]
nSurValue_m = [count_P1_m_nSur,count_P2_m_nSur,count_P3_m_nSur]


plt.bar(x,SurValue_m,width=width,color='red',label='Survived',alpha=0.5)
plt.bar(x+width,nSurValue_m,width=width,color='blue',label='Not Survived',alpha=0.5)
plt.xticks(x+width/3,Pclass)
plt.ylabel('Count')
plt.title('Pclass Barplot Gruoped by Survived in Modified Data')
plt.legend()
plt.show()
#%% NumFamily, SibSp, and Parch Plotting
TrainData['NumFamily'] = TrainRawData['SibSp'] + TrainRawData['Parch']
TestData['NumFamily'] = TestRawData['SibSp'] + TestRawData['Parch']
df_SibSp_Sur = TrainRawData['SibSp'].loc[TrainData['Survived']==1]
df_SibSp_nSur = TrainRawData['SibSp'].loc[TrainData['Survived']==0]
df_Parch_Sur = TrainRawData['Parch'].loc[TrainData['Survived']==1]
df_Parch_nSur = TrainRawData['Parch'].loc[TrainData['Survived']==0]

df_NF_Sur = TrainData['NumFamily'].loc[TrainData['Survived']==1]
df_NF_nSur = TrainData['NumFamily'].loc[TrainData['Survived']==0]

plt.hist(df_NF_Sur,alpha=0.7,label='Survived',bins = 6)
plt.hist(df_NF_nSur,alpha=0.3,label='Not Survived')
plt.title('Histogram of Number of Family')
plt.xlabel('Number of Family')
plt.ylabel('Count')
plt.legend()
plt.show()

plt.hist(df_SibSp_Sur,alpha=0.3,label='Survived',bins = 4)
plt.hist(df_SibSp_nSur,alpha=0.3,label='Not Survived',bins = 8)
plt.title('Histogram of SibSp')
plt.xlabel('SibSp')
plt.ylabel('Count')
plt.legend()
plt.show()

print(max(df_Parch_Sur))
print(max(df_Parch_nSur))
plt.hist(df_Parch_Sur,alpha=0.3,label='Survived',bins = 5)
plt.hist(df_Parch_nSur,alpha=0.3,label='Not Survived',bins = 9)
plt.title('Histogram of Parch')
plt.xlabel('Parch')
plt.ylabel('Count')
plt.legend()
plt.show()
#%% "Embarked" plotting
TrainData['Embarked'].describe()
df_Q = TrainData.loc[TrainData['Embarked'] == 'Q']
df_S = TrainData.loc[TrainData['Embarked'] == 'S']
df_C = TrainData.loc[TrainData['Embarked'] == 'C']

count_Q_nSur = len(df_Q.loc[df_Q['Survived'] == 0])
count_Q_Sur = len(df_Q.loc[df_Q['Survived'] == 1])
count_S_nSur = len(df_S.loc[df_S['Survived'] == 0])
count_S_Sur = len(df_S.loc[df_S['Survived'] == 1])
count_C_nSur = len(df_C.loc[df_C['Survived'] == 0])
count_C_Sur = len(df_C.loc[df_C['Survived'] == 1])

SurValue_emb = [count_Q_Sur,count_S_Sur,count_C_Sur]
nSurValue_emb = [count_Q_nSur,count_S_nSur,count_C_nSur]

Embarked = ['Q','S','C']
x = np.arange(len(Embarked))
width = 0.4
plt.bar(x,SurValue_emb,width=width,color='red',label='Survived',alpha=0.5)
plt.bar(x+width,nSurValue_emb,width=width,color='blue',label='Not Survived',alpha=0.5)
plt.xticks(x+width/3,Embarked)
plt.title('Barplot of Embarked Grouped by Survived')
plt.ylabel('Count')
plt.xlabel('Embarked')
plt.legend()
plt.show()
#%% Embarked Converting to Value
temp = []
for i in range(len(TrainData)):
    if TrainData['Embarked'].iloc[i] == 'Q':
        temp.append(0)
    elif TrainData['Embarked'].iloc[i] == 'S':
        temp.append(1)
    else:
        temp.append(2)
TrainData['EmbValue'] = temp

temp2 = []
for i in range(len(TestData)):
    if TestData['Embarked'].iloc[i] == 'Q':
        temp2.append(0)
    elif TestData['Embarked'].iloc[i] == 'S':
        temp2.append(1)
    else:
        temp2.append(2)
TestData['EmbValue'] = temp2
#%% LogFareLabel to make the Logfare discreted to run decision tree
# interval = 0.5
temp_LogFare_05 = []
for i in range(len(TrainData)):
    if TrainData['LogFare'].iloc[i] < 1:
        temp_LogFare_05.append(0)
    else:
        for j in range(1,12):
            if TrainData['LogFare'].iloc[i] >= (0.5+0.5*j):
                if TrainData['LogFare'].iloc[i] < (1.0+0.5*j):
                    temp_LogFare_05.append(j)
                    break
TrainData['LogFareLabel_05'] = temp_LogFare_05
# interval = 0.25
temp_LogFare_025 = []
for i in range(len(TrainData)):
    if TrainData['LogFare'].iloc[i] < 1:
        temp_LogFare_025.append(0)
    else:
        for j in range(1,23):
            if TrainData['LogFare'].iloc[i] >= (0.75+0.25*j):
                if TrainData['LogFare'].iloc[i] < (1.00+0.25*j):
                    temp_LogFare_025.append(j)
                    break
TrainData['LogFareLabel_025'] = temp_LogFare_025


tempTest_LogFare_05 = []
for i in range(len(TestData)):
    if TestData['LogFare'].iloc[i] < 1:
        tempTest_LogFare_05.append(0)
    else:
        for j in range(1,12):
            if TestData['LogFare'].iloc[i] >= (0.5+0.5*j):
                if TestData['LogFare'].iloc[i] < (1.0+0.5*j):
                    tempTest_LogFare_05.append(j)
                    break
TestData['LogFareLabel_05'] = tempTest_LogFare_05

tempTest_LogFare_025 = []
for i in range(len(TestData)):
    if TestData['LogFare'].iloc[i] < 1:
        tempTest_LogFare_025.append(0)
    else:
        for j in range(1,23):
            if TestData['LogFare'].iloc[i] >= (0.75+0.25*j):
                if TestData['LogFare'].iloc[i] < (1.00+0.25*j):
                    tempTest_LogFare_025.append(j)
                    break
TestData['LogFareLabel_025'] = tempTest_LogFare_025
#%% Save Data
#TrainData.to_csv('train_modified.csv',index=False)
#TestData.to_csv('test_modified.csv',index=False)
#%% Save Data in the different version
TrainData.to_csv('train_modified_v5.csv',index=False)
TestData.to_csv('test_modified_v5.csv',index=False)
