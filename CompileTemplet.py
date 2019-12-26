#%%
"""#!/usr/bin/env python
#title           :CompileTemplet.py
#description     :This will create a header for a python script.
#author          :gpandey
#date            :20190827
#version         :0.4 (updated times)
#usage           :python CompileTemplet.py
#notes           : All Iteration: it has all sort of modeling
#python_version  :3.7
#==============================================================================
"""
#Iteration1
#%% [Markdown]
## Put the topic tile

#%%
# Import warning Ignore
import warnings
warnings.filterwarnings('ignore')

#%%
# Set up your display size columna and row (but don't change until needed)
# Reconfiguring the display size to restrict figure size
pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 15)
# setting display option: shows total data 
pd.set_option('display.max_rows', None)

#%%
#Importing the required general packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
%matplotlib inline
import seaborn as sns

#%%
# Reading the Excel file 
df_raw=pd.read_excel('MA_Dis_Itr_1.xlsx')

#===========================================Python to SQL Server===================================
# connecting python to the SQL server to pull data from SQL Server
server= 'prd_75.sql.some.corp\prd_75'
db='EDWBronze'

#%%
# ODBC connection in order to pull out data from data from the refrenced server 
# and database 
import pyodbc
conn=pyodbc.connect('DRIVER={SQL Server}; SERVER=' +server+ '; DATABASE='+db+ '; Trusted_Connection=yes')
sql="""SELECT * FROM EDWBronze.cfa.Table1"""
sql2="""SELECT * FROM EDWBronze.cfa.Table2"""
df_Train=pd.read_sql(sql, conn)
df_Validation=pd.read_sql(sql2, conn)

####SQL server to R studio
install.packages("RODBC")
install.packages("dplyr")
library(RODBC)
library(dplyr)

conne <- 'Driver={SQL Server}; Server=prd_75.sql.some.corp\prd_75; Database=[EDWBronze];Uid=[Your_username];Pwd=[your_password];Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'
conn <- odbcDriverConnect(conne)
# or
conn<-odbcDriverConnect('driver={SQL Server};server=mysqlhost;database=mydbname;trusted_connection=true')

sqlcode <-"SELECT * FROM EDWBronze.cfa.cp_Train_test"
df <- sqlQuery(conn, sqlcode)

#%%
# copying data set incase needed for further use: save the original data
# (we are going to manipulate data from df within the same table)
df_raw=df.copy()
#%%
# counting unique member id ( to verify the correct form)
print('Unique MemberID TrainData: ',df_Train.MemberID.nunique())
print('Unique MemberID Validation Data: ',df_Validation.MemberID.nunique())

#%%
# Looking at the data
df_Validation.shape, df_Train.shape

#%% [markdown]
# **Data understanding**
#%%
df_raw.head()

#%%
df_raw.columns

#%%
# DropId columns
df=df_raw.drop('MEME_CK', axis=1)


#%%
####### Selecting Lastclaim Date to be only positive
# taking all the age bigger than 2
new_df=df[df.Age>=2]


#%%
# Data shape
df.shape

#%%
# all the info
df.info()
#Looking all the data types with not null counts: Verbose= True, null-counts=True give the details of info
# like data types, count of null/not null values
df.info(verbose=True, null_counts=True)
#%%
# Null values
df.isnull().sum()
#%%
print("Total number of null in the whole dataset")
print(df.isnull().sum().sum())

#%% 
# Droping Dublicate columns, and noise data (extra unnecessary columns)
df.drop([ 'TransTypeCode', 'TransReplyCode'	],axis=1, inplace=True)

#%%
# listing one value from one column to other
df.insert(loc=50, column='Columnname', value=df['LAST_CL_DT'].sub(df['FIRST_CL_DT'], axis=0))

#############Flagging based on column values of different column###############
#Creating one column based on values of other column
import numpy as np
# Create a new column called data.added where the value is yes
# if df.age is greater than 50 and no if not
data['added'] = np.where(data['Age']>=1000, 'yes', 'no');
data['name_math'] = data['added'].apply(lambda x: 56 if x == 'yes' else 80)
# This function creates the a new columns in the the table based on previous two or three columns
def flag_df(dfa):

    if (dfa['StayType'] == 'Admission')  and (dfa['name_math'] ==56) and (dfa['added'] =='yes'):
        return 'don\'t know'
    elif (dfa['StayType'] == 'Admission')and (dfa['name_math'] ==80):
        return 'Kind of expected'
    elif (dfa['StayType'] == 'Reentry')  and (dfa['name_math'] ==56):
        return 'souns good'
    else:
        return np.nan

data['Flag'] = data.apply(flag_df, axis = 1)
data.head()

# This function creates new columns in a table asking if the flag column is
#  null then sum to otherwise keep it as it is
def flag_df1(dfb):

    if (dfb['Flag'] == 'Null'):
        return dfb['MDSEpisodLength']+dfb['name_math']
    else:
        return dfb['Flag']

data['Newval'] = data.apply(flag_df1, axis = 1)
data.head()

#Adding two values
data.loc[data['Flag'].isnull(),'Flag'] = data['MDSEpisodeLength']+data['name_math']
data.head()
#%%
df.info()


#%%
df.head()

################# Renaming the column in dataframe####################
# Let say in data there is name sales and want to change for dept then
data = pd.read_csv('../input/HR_comma_sep.csv')
data = data.rename(columns={'sales': 'dept'})

#%%
###############Converting date to days#######################
# Converting the date time for 'somethingdate'
df['somethingdate'] = df['somethingdate'] / np.timedelta64(1, 'D')

#%%
############### Date difference################
from datetime import date

def diff_dates(date1, date2):
    return abs(date2-date1).days

#%%
# dropping the First and Last date
df.drop(['firstdate','Lastdate'], axis=1, inplace=True)

#%%
#==========================================================Feature Engineering===========================================
# Get the column index to insert new column
df.columns.get_loc('somecolumn')

#%%
#Computing the average payment, Avg patient paid amt and average claim count
df.insert(loc=30, column='Newcolumnname', 
           value=(((df1['somecolumn1']/df1['sm1'])+(df1['somecolumn2']/df1['sm2']))+
                   (df1['somecolumn3']/df1['sm3']))/3)

#%% [markdown]
################# Null Value Analysis##############################
#Missing data

# -Important questions when thinking about missing data:


# - How prevalent is the missing data?

#- Is missing data random or does it have a pattern?  -The answer to these questions is 
# important for practical reasons because missing data can imply a 
#  reduction of   the sample size. This can prevent us from proceeding with the analysis. 
#   Moreover, from a substantive perspective, we need to ensure that the missing data process 
#   is not biased and hidding an inconvenient truth


#%%
######################Anu
# getting all null values 
total = df.isnull().sum().sort_values(ascending=False)
percent=(df.isnull().sum()/df.isnull().count()*100).sort_values(ascending=False)
Missing_data=pd.concat([total, percent], axis=1, keys=["Total Mising", "Percent Missing"])
# Print the result
Missing_data
#%%
# drop all the values where outcome variable is null
df=df.dropna(subset=['target'])

#%%
# Printing the unique values 
print("unique values of column1", df.column1.unique())
print("unique values of column2", df.column2.unique())
print("unique values of ElectionType", df.ElectionType.unique())

#%%
# looking the unique values with frequencies
df.column1.value_counts()
df.column1.value_counts()

#%%
#%% [markdown]
# Data profiling: maintaining the consistancy by converting data into correct format
# - Converting categorical feature into categories and numerical features in numeric
# -  status is Categorical =covert to category (int to category)
# - id into category (int to category)
# - electiontype to category (int to category)

#%%
# Converting to categorical
df['columns1'] = pd.Categorical(df['columns1'])
df['columns2'] = pd.Categorical(df['columns2'])
# converting to object
df['columns2'] = df['columns2'].astype(str)
df['MonthOflove']=df['MonthOflove'].astype(str)

#%%
## Text processing: converting into lowercase for all features values with text type
# - We saw there are two types 'yes' or 'no' but became 4 categories on married 
#%% 
# convert all text type feature values into lower case
df['Married']= df['Married'].str.lower()
df['Hasfamily']=df['Hasfamily'].str.lower()

#%%
###############Segreggation of data #########################
# segregation of continous and categorical data
df_cate=df1.select_dtypes(include=['object', 'category'])
df_cont=df1.select_dtypes(exclude=['object','category']

#OR 
# Continous features without target
confeatures_notarget = [col for col in df_cont.columns if col!='Voluntary_disenroll']
# droping all the continous features
dropcol=list(confeatures_notarget)
CateDatawithTarget=df1.drop(dropcol, axis=1)

# other method to get categorical with target
to_drop=ContDf.columns
df_catewithTarget=df[df.columns.difference(to_drop)]
df_catewithTarget.shape
#select categorical features without target variable
cate_featurenoTarget=[col for col in CateDatawithTarget.columns 
                      if col!='Voluntary_disenroll']

# drop the target variable
ContDf=df_cont.drop('Voluntary_disenroll', axis=1)
# Histogram for all continuous variables
ContDf.hist(figsize=(20,30), bins=10, xlabelsize=3, ylabelsize=4);


#%%
#%% [markdown]
#### Looking the basic statistics of the continous variables
# describe the 
df.describe()
#%%[markdown]
#There are three aspects that usually catch our attention when we analyse descriptive statistics:

# - Min and max values. This can give us an idea about the range of values and is helpful to detect 
# outliers. In our case, all the min and max values seem reasonable and in a reasonable range of values. 
# The only exception could eventually be the max value of 'Maxepisodelength', but for now we will 
# leave it as it is. 


# - Mean and standard deviation. The mean shows us the central tendency of the distribution, 

#- while the standard deviation quantifies its amount of variation. For example, a low 
# standard deviation suggests that data points tend to be close to the mean. Giving a quick 
# look to our values, there's nothing that looks like obviously wrong. -Count. This is important 
# to give us a first perception about the volume of missing data. 

# - Here, we can see that some 'Age' data is missing and other several variable as we saw before.

#%%
############Null Values imputation
# continous
df['continuous1'].fillna(df.continuous1.mean(), inplace=True)
df['continuous2'].fillna(df.continuous2.median(), inplace=True)
# Fill missing values in all categorical features  with a specific value that we mentioned before
# if you want to replace null with highest frequent for whole data set
# - df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
# otherwise
df['catecol1'].fillna(df.catecol1.value_counts().index[0], inplace=True)
df['catecol2'].fillna(df.catecol2.value_counts().index[0], inplace=True)

#%%
# statistics  only for continuous variables
# Again looking up basic statistics
print("Statistics for Only Continous variables")
df.describe() 

#%%
# Statistics: only for object variables

print("Statistics for Only object variables")
df.describe(include=[np.object])  
#%%
# Statistics: only for Categorical variables

print("Statistics for Only Categorical variables")
df.describe(include=['category'])

#%%
# Statistics: only for object and Categorical variables

print("Statistics for Only Categorical and Object -not_numeric variables")
df.describe(exclude=[np.number])  


#%%
# Statistics: for all variables

print("Statistics for Only Categorical and Object -not_numeric variables")
# descriptive statistics for all
df.describe(include='all')


#%%
################################################################################
######################## EDA-Explanatory Data Analysis#############################
##################################################################################

#%%
###############Target variable plot #######################
###Target value bar plot with hight
# counting the successful discharge
df.Survived.value_counts()
# Count plot of target variables
sns.countplot(x="Survived", data=df, palette="bwr")
plt.show()
## code for getting with the height
# Count plot of target variables with height
plt.figure(figsize=(9,8))
ax = sns.countplot(x="SUCCESSFUL_DISCHARGE", data=df5, palette="bwr")
plt.title('Distribution of Discharge Status')
plt.xlabel('Discharge Status')
plt.ylabel('Count')


# get the height for your bar plot
for p in ax.patches:
        ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+50))

# Percentwise display of the target variable
count_SD = len(df[df.survived == 1])
count_NSD = len(df[df.survived == 0])
print("Percentage of passenger survived: {:.2f}%".format((count_SD / (len(df.survived))*100)))
print("Percentage of passenger not survived: {:.2f}%".format((count_NSD / (len(df.survived))*100)))

# Percentwise plot
## Percentwise display of the target variable with height in percent
plt.figure(figsize=(12,8))
plt.title('Distribution of Discharge Status')
plt.xlabel('Discharge Status')
plt.ylabel('Count')

ax = (df5.SUCCESSFUL_DISCHARGE.value_counts()/len(df5)*100).sort_index().plot(kind="bar", rot=0)
ax.set_yticks(np.arange(0, 110, 10))

ax2 = ax.twinx()
ax2.set_yticks(np.arange(0, 110, 10)*len(df5)/100)

for p in ax.patches:
    ax.annotate('{:.2f}%'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))

#################### Looking at the unique values
df.sex.unique()
df.sex.value_counts()

#################Genderwise scatter plot of one continuous variable Age
#Scatter plot for genderwise 
plt.scatter(x=df.Age[df.Survived==1], y=df.familysize [(df.survided==1)], c="red")
plt.scatter(x=df.Age[df.Survived==0], y=df.familysize [(df.survived==0)])
plt.legend(["Survided", "NSurvived"])
plt.xlabel("Age")
plt.ylabel("familysize")
plt.show()
########################Bar plot for one continuos and cate target variable
#Visualization of Survived for Ages
pd.crosstab(df5.Age,df.Survived).plot(kind="bar",figsize=(20,6))
plt.title('Survived Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('SruvivedAndAges.png')
plt.show()
##############Genderwise survive plot
# cross Checking of the 
pd.crosstab(df.Gender,df.Survived).plot(kind="bar",figsize=(15,6),
                                         color=['#AA1111', '#1CA53B' ])
plt.title('SD Frequency for Sex')
plt.xlabel('Gender ( 1 = Male, 2 = Female)')
plt.xticks(rotation=0)
plt.legend(["Survived", "NotSurvived"])
plt.ylabel('Frequency')
plt.show()

#######################bar plot
# Bar plot for pclass
sns.barplot(df['Plcass'],df['survived']);
#%%
# Histogram of only Numerical values
import matplotlib.pyplot as plt
df[df.dtypes[(df.dtypes=="float64")|(df.dtypes=="int64")].index.values].hist(figsize=[11,11])

#Distribution percent of target variable
df.Target.value_counts()/df.Target.count()
#%%
# Group by plot: hist of contifeatures based on target category
df.groupby('targetvari').contifeature1.hist();   
# counting of genderwise 
df.groupby(['Gender'])['pclass'].count()         
#%%
# cross Checking of the categoricla with target
plt.figure(figsize=(8,5))
pd.crosstab(df.catecol1,df.target).plot(kind="bar",figsize=(8,6),color=['#1CA53B', '#AA1111'])
plt.title('catecol1')
plt.show()

# cross Checking of the member contact attem continuous vs target
plt.figure(figsize=(8,5))
pd.crosstab(df1.conticol1,df1.target).plot(kind="bar",figsize=(8,6),color=['#1CA53B', '#AA1111'])
plt.title('conticol1')
plt.show()
#%%
# Box plot for continus variable w.r.t cate target variable
plt.ylim(-10, 50)
sns.boxplot(x="Target_cate",y="predictorvar_cont" , data=Data_Cont)
# bar plot for Gender
sns.countplot(x='Sex', data=df);

#%%
# Voluntary Disenroll by Gender (note: This only creates disenrolled )
df.groupby(['Sex']).Target.sum()
# Looking at the the genderwise average age and average stay length for the transition status
df.groupby(['Gender', 'Survived']).mean()
# Data Analysis based on gender, successful_discharge, HipFracture, Shortterm memory and averge age
data_grop=df.groupby(['Gender', 'Sruvived', 
                              'pclass', 'embark']).mean()

##############target with gender percent

## voluntary Disenroll
print("Female Disenroll Percent", df[df.Sex == 'F'].Voluntary_disenroll.sum()/df[df.Sex == 'F'].Voluntary_disenroll.count()*100)
print("Male Disenroll Percent", df[df.Sex == 'M'].Voluntary_disenroll.sum()/df[df.Sex == 'M'].Voluntary_disenroll.count()*100)
#####
sns.factorplot(x='Voluntary_disenroll', col='Sex', kind='count', data=df);

#%%
sns.factorplot(x='Voluntary_disenroll', col='Inst_NHC', kind='count', data=df);
#%%
# Catplot for Multi-variate analysis
sns.catplot(x="Gender", y="Survived",  hue="pclass", 
            col="embark", data=df, kind="bar", height=4, aspect=.7);
# count plot
sns.countplot(x='catecol1', hue='Target', data=df1);
#%%
# Count plot of target variables with height
plt.figure(figsize=(9,8))
ax = sns.countplot(x='catecol1',hue='Voluntary_disenroll', data=df1, palette="bwr")
plt.title('Distribution of Voluntary_disenroll')
plt.xlabel('Voluntary_disenroll')
plt.ylabel('Count')
# get the height for your bar plot
for p in ax.patches:
        ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+1))

#%%
##################Correlation and Scatter plot betn two variables#################
# Checking the correlation of TotalMedicalClaim and Pharmacy claim Scatter plot
x=df_cont['Conticol1']
y=df_cont['Contcol1']
plt.scatter(x, y);

#%%
# Analysis of correlation between all the continous features with Conticol1t
features = [col for col in df_cont.columns if col!='Conticol1t']
df_cont_corr=df_cont.corr()['Conticol1t'][features]
zist_features_list=df_cont_corr[abs(df_cont_corr)>0.2].sort_values(ascending=False)
print("There is {} strongly coreelated values with gross: \n{}"
      .format(len(zist_features_list), zist_features_list))
#%%
#################Histogram for continuous variable###################
import math
fig = plt.figure(figsize=(20,15))
cols = 5
rows = math.ceil(float(Data_Cont.shape[1]) / cols)
for i, column in enumerate(Data_Cont.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if Data_Cont.dtypes[column] == np.object:
        Data_Cont[column].value_counts().plot(kind="box")
    else:
        Data_Cont[column].hist(axes=ax)
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2)
plt.show()

#%%
############hist plot#########
df['familymember'].hist();
#%%[markdown]
#######################Correlation########################
## Correlation-heatmap
# only for Continuous variables
#%%
# correlation between numeric column
#correlation matrix
corrmat = df.corr()
f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(corrmat, vmax=1.0, vmin=-1.0, square=True);

#%%
# heatmap with corerlation coefficeint value presenting to the table( annnotate give the values of corr coeff.)
colormap=plt.cm.plasma
plt.figure(figsize=(10,10))
plt.title('Correlation of features and Target', y=1.05, size=14)
sns.heatmap(df.corr().round(2), linewidth=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True);

#%%
#Heat map plotting w.r.t target variable
sns.set(font_scale=1.0)
plt.figure(figsize=(10,10))
sns.heatmap(df.corr()[['Target']], annot=True);

#%% 
# Check for multicollinarity by dropping target
corr=df.drop('Target', axis=1).corr()
plt.figure(figsize=(12,12))
sns.heatmap(corr[(corr>=0.35)| (corr<=-0.35)], cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);

#####################Correlation in order #################
# Value display
# multi-colinarity plot via heat map with high correlated display
corr=df_cont.drop('target', axis=1).corr()
plt.figure(figsize=(20,20))
sns.heatmap(corr[(corr>=0.75)| (corr<=-0.75)], cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 4}, square=True)

# ContDf is the continous features values
# looking at the correlation 
ContDf.corr().abs().unstack().sort_values(ascending=False).drop_duplicates()
##########################
#Again Checking the correlation between features
corr2 = df_cont_removed.corr() 
c1 = corr2.abs().unstack()
c1.sort_values(ascending = False).drop_duplicates()
########################Heat map with Target variable######################
#Heat map plotting w.r.t target variable
plt.figure(figsize=(10,10))
sns.heatmap(df1.astype(float).corr()[['Targetvariable']], vmax=1.0, vmin=-1.0, annot=True);

## Find all correlations with the score and sort them 
correlations_data = data.corr()['targetvarib_cate'].sort_values(ascending=False)
##############Multicolinarity test######################################
# Heat map for multicolinarity 
corr=df1.drop('Cate_target', axis=1).astype(float).corr()
plt.figure(figsize=(20,20))
sns.heatmap(corr[(corr>=0.35)| (corr<=-0.35)], cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);
#%%
#############################Cramers-V function###########################
# Correlation for categorical features using Cramer's V function
# get all the categorical data
df_cate=df.select_dtypes(include=['object'])
#%%
# checking out the colinarity of categorical features 
# using Cramers-V function

def cramers_v(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

Corr_out=[]
factors_paired=[(i,j) for i in df_cate.columns.values 
for j in df_cate.columns.values]
for f in factors_paired:
    if f[0]!=f[1]:
        confusion_matrix = pd.crosstab(df_cate[f[0]], df_cate[f[1]]).as_matrix()
        Corr_out.append(cramers_v(confusion_matrix))
    else:
        Corr_out.append(0)
Corr_out=np.array(Corr_out).reshape((36,36))
Cramer=pd.DataFrame(Corr_out,index=df_cate.columns.values,columns=df_cate.columns.values)

#%%
# Export the  data to Excel file to check and verify
Cramer.to_excel("CateCorrelation.xlsx")
# SNS plot for heat map of categorical for cramers function
sns.heatmap(Cramer)


#%%
######################Chi-square test#######################
#%%
df_cate=df.select_dtypes(include=['object'])
# For categorical variables
import scipy.stats as stats
from scipy.stats import chi2_contingency

class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        
        self.dfTabular = None
        self.dfExpected = None
        
    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p<alpha:
            result="{0} is IMPORTANT for Prediction".format(colX)
        else:
            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)

        print(result)
    def TestIndependence(self,colX,colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        
        self.dfObserved = pd.crosstab(Y,X) 
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        
        self._print_chisquare_result(colX, alpha)

#%%
# apply the defined function
#Initialize ChiSquare Class
cT = ChiSquare(df_cate)

#Feature Selection
testColumns = cate_featurenoTarget
for var in testColumns:
    cT.TestIndependence(colX=var,colY='Voluntary_disenroll' ) 
#%%
#####################Chi-Square test#########################
import scipy.stats as stats
from scipy.stats import chi2_contingency
#%%
#Checking the significance for attribution and Assign desc
contingency_table=pd.crosstab(index=df1.Attribution_Type,columns=df1.Attribution_Type)
contingency_table

#%%
#chi2_contingency(contingency_table)[0:3]
chi2_contingency(pd.crosstab(df1.Attribution_Type, df1.ASSIGN_DESC))[0:3]

#%%
chi2_contingency(pd.crosstab(df1.Attribution_Type, df1.SameAssignAttrProv))[0:3]

#%%
###########################################ANoVA#############################
# ANOVA (one-way)
stats.f_oneway(df1.LastClaim_Disenrl_Duration, df1.Voluntary_disenroll)

#%%
sns.catplot(x="Voluntary_disenroll", y="LastClaim_Disenrl_Duration", kind="box", data=df1);

#%%
sns.catplot(x="Voluntary_disenroll", y="LastClaim_Disenrl_Duration", kind="bar", data=df1)

#Group by
df1.groupby('Voluntary_disenroll').agg([np.mean, np.median,  np.std]).MemberEnrlDurInMonths
#%%
####################Simple mapping values to number##############
# lets map Sex: male to 0 and female to 1
mapping = {'male':0,
          'female':1}
data_train['Sex'] = data_train['Sex'].map(mapping)

#####################################ONe Hot Encoding#####################
# One Hot Encoding
# get dummies for categorical and object
data = pd.get_dummies(df)  

############################# Label Encoding #######################
# Importing LabelEncoder and initializing it
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
# Iterating over all the common columns in train and test
for col in X_test.columns.values:
       # Encoding only categorical variables
       if X_test[col].dtypes=='object':
       # Using whole data to form an exhaustive list of levels
       data=X_train[col].append(X_test[col])
       le.fit(data.values)
       X_train[col]=le.transform(X_train[col])
       X_test[col]=le.transform(X_test[col])
#######################Other method of Label encoding ####################
from sklearn.preprocessing import LabelEncoder

class preproc():
    
    def __init__(self, data, cols):
        self.data = data
        
    def transform(self, dummies=False):
        if dummies:
            print('getting dummies for cat. variables...')
            self.data = pd.get_dummies(self.data, columns=cols)
            return self.data
        else:
            for col in cols:
                print('label encoding...')
                le = LabelEncoder()
                le.fit(self.data[col])
                self.data[col] = le.transform(self.data[col]) 
                print(le.classes_)
            return self.data
#call the above function
cols = ['dept', 'salary']
pp = preproc(data, cols)
data = pp.transform(dummies=False)

#%%[markdown]
##################Data Transformation##############################
# Data transformation usually is done after dummies before spliting train and test
#you can do either way
# max min transformation
df3=(data - np.min(data)) / (np.max(data) - np.min(data)).values

#%%
# Segredation of explanatory and Target variables
x_data = df3[df3.loc[:, df3.columns != 'Voluntary_disenroll'].columns]
y = df3['Voluntary_disenroll']


#%% 
#########################OR##########################3
# Spliting data into target and explanatory
x_data = data[data.loc[:, data.columns != 'Target'].columns]
y = data['Target']

#%%
# Data transformation --scaled all features into similar scale
#Max min Transformation
X = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

#%%
#############################Modeling#######################################
#==================================================Modeling==========================================================
# importing different packages for ML modeling
from sklearn.linear_model import LogisticRegression
#validation and learning curver
from sklearn import metrics
from sklearn.metrics import confusion_matrix, recall_score, precision_score, precision_recall_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
#test train split
from sklearn.model_selection import train_test_split
#importing matrics 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
#ROC curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

#%%
# Train and Test split of data 80-20 
x_train, x_test, y_train, y_test = train_test_split(x_data,y,test_size = 0.2,random_state=0)

#%%
# Data transformation standard scaler and Max-Min transformation 
#  (it has to be done after split)
df_con=df.select_dtypes(exclude=['object','category'])
###############Max-Min Transformation ##########################
from sklearn.preprocessing import MinMaxScaler
min_max=MinMaxScaler()

X_train_minmax=min_max.fit_transform(x_train[['ApplicantIncome', 'CoapplicantIncome',
                'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
X_test_minmax=min_max.fit_transform(x_test[['ApplicantIncome', 'CoapplicantIncome',
                'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])

# Or
scaler = MinMaxScaler(feature_range = (0,1))

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Fitting k-NN on our scaled data set
>> knn=KNeighborsClassifier(n_neighbors=5)
>> knn.fit(X_train_minmax,Y_train)
# Checking the model's accuracy
>> accuracy_score(Y_test,knn.predict(X_test_minmax))
####################### Standard Scaling  ###################
Standardizing the train and test data
from sklearn.preprocessing import scale
X_train_scale=scale(X_train[['ApplicantIncome', 'CoapplicantIncome',
                'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
X_test_scale=scale(X_test[['ApplicantIncome', 'CoapplicantIncome',
               'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
# Fitting logistic regression on our standardized data set
from sklearn.linear_model import LogisticRegression
log=LogisticRegression(penalty='l2',C=.01)  # Here penalty=l1 or l2 means regularization
log.fit(X_train_scale,Y_train)
# Checking the model's accuracy
accuracy_score(Y_test,log.predict(X_test_scale))

## Standar scaling another method
# Lets drop target variable
df_cont_filledNotarget=df_cont_filled.drop('Voluntary_disenroll', axis=1)

# standard scaling data
from sklearn import preprocessing
names = df_cont_filledNotarget.columns
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df_cont_filledNotarget)
scaled_df = pd.DataFrame(scaled_df, columns=names)
# then you can devide into train and test


#%%
############################## Balancing the imbalanced dataset ############
from imblearn.over_sampling import SMOTE

X_train, X_test, y_train, y_test = train_test_split(X_df_2, y, test_size = 0.2, random_state = 10)

sm = SMOTE(random_state = 2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

clf = LogisticRegression()
model_res = clf.fit(X_train_res, y_train_res)

#%%
#===============================Different Model Application============================
# Fitting different Models and comparing the performance matric
# checking with logistic regression from sklearn
lr = LogisticRegression();
lr.fit(x_train,y_train);
print("Test Accuracy {:.2f}%".format(lr.score(x_test,y_test)*100));

#%%
#CrossValidation score
scores = cross_val_score(lr, x_train, y_train, cv=10)
print('CV accuracy (original): %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

#%%


#%%
# Predicted values
y_pred_lr = lr.predict(x_test)
cm_lr = confusion_matrix(y_test,y_pred_lr)
plt.title("Logistic Regression Confusion Matrix")
sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False);

#%%
# Just looking at the accuracy (not crossvalidation)
from sklearn import metrics
y_pred= lr.predict(x_test)
print(metrics.accuracy_score(y_pred,y_test))

#%%
# For single class:
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# we have to give y_test and y_predicted values for ROC curve


fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label = 1)

roc_auc = auc(fpr, tpr)

plt.figure(1, figsize = (8, 8))
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, marker='o', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.show()

#%%[markdown]
## Fitting Different models


#%%
# Fitting the Naive Bayes Classificatin
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)
print("Accuracy of Naive Bayes: {:.2f}%".format(nb.score(x_test,y_test)*100))


#%%
# Fitting the KNN model
# KNN Model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)

print("{} NN Score: {:.2f}%".format(2, knn.score(x_test, y_test)*100))

#%%
# try ro find best k value
scoreList = []
for i in range(1,20):
    knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k
    knn2.fit(x_train, y_train)
    scoreList.append(knn2.score(x_test, y_test))
    
plt.plot(range(1,20), scoreList)
plt.xticks(np.arange(1,20,1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()


print("Maximum KNN Score is {:.2f}%".format((max(scoreList))*100))

#%%
# Lets try different models SVM
from sklearn.svm import SVC
svm = SVC(random_state = 1)
svm.fit(x_train, y_train)
print("Test Accuracy of SVM Algorithm: {:.2f}%".format(svm.score(x_test,y_test)*100))
#%%
# Apply the Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
print("Decision Tree Test Accuracy {:.2f}%".format(dtc.score(x_test, y_test)*100))

#%%
# Applying the Random Forest
# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
rf.fit(x_train, y_train)
print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(rf.score(x_test,y_test)*100))

#%%
# ROC Curve for Random Forest
#%%
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot 
# roc curve plotting
probs=rf.predict_proba(x_test)
probs=probs[:, 1]

#%%
# calculate AUC
auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)
#%%
# calculate roc curve
fpr1, tpr1, thresholds1 = roc_curve(y_test, probs, pos_label = 1)
#%%
from sklearn.metrics import auc
roc_auc1 = auc(fpr1, tpr1)
#%%
# Plotting the ROC curve
pyplot.plot([0, 1], [0, 1], linestyle='--')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.title('Receiver operating characteristic curve')
# plot the roc curve for the model
pyplot.plot(fpr1, tpr1, marker='.',label='ROC curve (area = %0.2f)' % roc_auc1)
pyplot.legend(loc="lower right")
# show the plot
pyplot.show()

#%%

#%%[markdown]
## Comparing the model Performances
methods = ["Logistic Regression", "KNN", "SVM", "Naive Bayes", "Decision Tree", "Random Forest"]
accuracy = [98.5, 97.76, 98.13,96.5, 98.04, 98.79]
colors = ["purple", "green", "orange", "magenta","#CFC60E","#0FBBAE"]

sns.set_style("whitegrid")
plt.figure(figsize=(10,5))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=methods, y=accuracy, palette=colors)
plt.show()

#%%[markdown]
## Combining confusion Matrix
#%%
y_head_knn = knn.predict(x_test)
y_head_svm = svm.predict(x_test)
y_head_nb = nb.predict(x_test)
y_head_dtc = dtc.predict(x_test)
y_head_rf = rf.predict(x_test)

#%%
# Creating the confusing matrix
cm_lr = confusion_matrix(y_test,y_head_lr)
cm_knn = confusion_matrix(y_test,y_head_knn)
cm_svm = confusion_matrix(y_test,y_head_svm)
cm_nb = confusion_matrix(y_test,y_head_nb)
cm_dtc = confusion_matrix(y_test,y_head_dtc)
cm_rf = confusion_matrix(y_test,y_head_rf)

#%%
# Confusion Matrix plot
# Confusion matrix plot
plt.figure(figsize=(16,12))

plt.suptitle("Confusion Matrixes",fontsize=24)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

plt.subplot(2,3,1)
plt.title("Logistic Regression Confusion Matrix")
sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False)

plt.subplot(2,3,2)
plt.title("K Nearest Neighbors Confusion Matrix")
sns.heatmap(cm_knn,annot=True,cmap="Blues",fmt="d",cbar=False)

plt.subplot(2,3,3)
plt.title("Support Vector Machine Confusion Matrix")
sns.heatmap(cm_svm,annot=True,cmap="Blues",fmt="d",cbar=False)

plt.subplot(2,3,4)
plt.title("Naive Bayes Confusion Matrix")
sns.heatmap(cm_nb,annot=True,cmap="Blues",fmt="d",cbar=False)

plt.subplot(2,3,5)
plt.title("Decision Tree Classifier Confusion Matrix")
sns.heatmap(cm_dtc,annot=True,cmap="Blues",fmt="d",cbar=False)

plt.subplot(2,3,6)
plt.title("Random Forest Confusion Matrix")
sns.heatmap(cm_rf,annot=True,cmap="Blues",fmt="d",cbar=False)

plt.show()

#%%
# Plotting all the ROC Curver together
#%%
#==================================ROC Curve plotting together=============================
# Computation of AUC for all 3 Models
probs=lr.predict_proba(x_test)
probs=probs[:, 1]
probs1=dtc.predict_proba(x_test)
probs1=probs1[:, 1]
probs2=rf.predict_proba(x_test)
probs2=probs2[:, 1]


#%%
# plotting the Roc Curve together
# pos_label are the values in target so it if it object it should be pos_label = '1' if int pos_label = 1
from sklearn.metrics import auc
plt.figure(0).clf()
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curves')
fpr, tpr, thresholds = roc_curve(y_test, probs, pos_label = 1) # pos_label are the values in target 
auc1 = roc_auc_score(y_test, probs)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, marker='.',label='ROC curve_logreg (area = %0.4f)' % roc_auc)

fpr1, tpr1, thresholds1 = roc_curve(y_test, probs1, pos_label = 1)
auc2 = roc_auc_score(y_test, probs1)
roc_auc1 = auc(fpr1, tpr1)
plt.plot(fpr1, tpr1, marker='.',label='ROC curve_DTree (area = %0.4f)' % roc_auc1)

fpr2, tpr2, thresholds2 = roc_curve(y_test, probs2, pos_label = 1)
auc3 = roc_auc_score(y_test, probs2)
roc_auc2 = auc(fpr2, tpr2)
plt.plot(fpr2, tpr2, marker='.',label='ROC curve_RForest (area = %0.4f)' % roc_auc2)

plt.legend(loc=0)

#######other subplot of all roc together
# plotting all the Roc Curve Together
# Logistic Regression
from sklearn.metrics import auc
probs1=lr.predict_proba(x_test)
probs1=probs1[:, 1]
# calculate AUC
auc1 = roc_auc_score(y_test, probs1)
print('AUC: %.3f' % auc1)
# calculate roc curve
fpr1, tpr1, thresholds1 = roc_curve(y_test, probs1, pos_label = '1')

# Decision Tree
probs2=dtc.predict_proba(x_test)
probs2=probs2[:, 1]
# calculate AUC
auc2 = roc_auc_score(y_test, probs2)
print('AUC: %.3f' % auc2)
# calculate roc curve
fpr2, tpr2, thresholds2 = roc_curve(y_test, probs2, pos_label = '1')


# Random Forest
probs3=rf.predict_proba(x_test)
probs3=probs3[:, 1]
# calculate AUC
auc3 = roc_auc_score(y_test, probs3)
print('AUC: %.3f' % auc3)
# calculate roc curve
fpr3, tpr3, thresholds3 = roc_curve(y_test, probs3, pos_label = '1')

#%%
# computing Area under the curve for all three
roc_auc1 = auc(fpr1, tpr1)
roc_auc2 = auc(fpr2, tpr2)
roc_auc3 = auc(fpr3, tpr3)

#%%
# Roc Curve for all three plot
plt.figure(figsize=(16,12))

plt.suptitle("ROC Curve Comparision",fontsize=24)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)


plt.subplot(2,3,1)
plt.title(" ROC for Logistic Regression")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plot the roc curve for the model
plt.plot(fpr1, tpr1, marker='.',label='ROC curve (area = %0.2f)' % roc_auc1)
plt.legend(loc="lower right")


plt.subplot(2,3,2)
plt.title("ROC for Decision Tree")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plot the roc curve for the model
plt.plot(fpr2, tpr2, marker='.',label='ROC curve (area = %0.2f)' % roc_auc2)
plt.legend(loc="lower right")

plt.subplot(2,3,3)
plt.title(" ROC for Random Forest ")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plot the roc curve for the model
plt.plot(fpr3, tpr3, marker='.',label='ROC curve (area = %0.2f)' % roc_auc3)
plt.legend(loc="lower right")

plt.show()
#%%
#######################Precision and Recall Score#########################
# Printing Precision and Recall score together
# Printing all the recall together in order to make decsion on performance matric
from sklearn.metrics import confusion_matrix, recall_score, precision_score, precision_recall_curve
print("Logistic Regression Recall Score is:", recall_score(y_test, y_head_lr, pos_label=1))
print("Decision Tree Recall Score is:", recall_score(y_test, y_head_dtc, pos_label=1))
print("Random Forest Recall Score is:", recall_score(y_test, y_head_rf, pos_label=1))
# Precision  Score for Decision Tree and Random forest
print('The precision score for Decision Tree: ', precision_score(y_test, y_head_dtc, pos_label=1))
print('The precision score for Random Forest: ', precision_score(y_test, y_head_rf, pos_label=1))


#%%
#################importand feature selection###################################
################################################################################
# you have to apply this method for the data after get dummies but before 
# train test split: in the following data
data=pd.get_dummies(df, drop_first=True)

# Important Feature Selection using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier 
model= RandomForestClassifier(n_estimators=100,random_state=0)
Xx=data[data.loc[:, data.columns != 'Voluntary_disenroll'].columns]
Yy=data['Voluntary_disenroll']
model.fit(Xx,Yy)
pd.Series(model.feature_importances_,index=Xx.columns).sort_values(ascending=False)
## Visualization of those important features
plt.figure(figsize=(8,10))
feat_importances = pd.Series(model.feature_importances_, index=Xx.columns)
feat_importances.nlargest(30).plot(kind='barh')
plt.show()

#%%[markdown]
# Plotting the important features besed on highest to lowest Random Forest

#%%
# Applying the Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100, random_state = 2)
rf.fit(x_train, y_train)
print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(rf.score(x_test,y_test)*100))

#%%
# Get the Highest predictors
pd.Series(rf.feature_importances_,index=x_train.columns).sort_values(ascending=False)

#%%
plt.figure(figsize=(8,10))
plt.ylabel("Features")
plt.xlabel("Importance ")
feat_importances = pd.Series(rf.feature_importances_, index=x_train.columns)
#feat_importances.nlargest(20).plot(kind='barh')
#plt.show()
# to print the plot in opposite order
feat_importances1=feat_importances.to_frame().reset_index()
feat_importances1.columns=['Feature','Importance']
sort_by_importance=feat_importances1.sort_values('Importance',ascending=False)
top_20_feat=sort_by_importance.head(20)
top_20_feat=top_20_feat.reset_index(drop=True)
plt.gca().invert_yaxis()
plt.barh(top_20_feat['Feature'],top_20_feat['Importance'])
plt.rc('xtick',labelsize=6)
plt.rc('ytick',labelsize=6)

#%%
# Important Features from decision tree and Decision tree plot
#Feature importance from Decision tree
pd.Series(dtc.feature_importances_,index=Xx.columns).sort_values(ascending=False)
#%%
plt.figure(figsize=(8,10))
plt.ylabel("Features")
plt.xlabel("Importance %")
feat_importances = pd.Series(dtc.feature_importances_, index=Xx.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

############important feature 4th techniques#####################
##########by using extra tree classifier
from sklearn.ensemble import ExtraTreesClassifier
plt.figure(figsize=(20,20))
model = ExtraTreesClassifier()
model.fit(Xx,Yy)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=Xx.columns)
feat_importances.nlargest(30).plot(kind='barh')
plt.show()
################ important features selection 5th techniques############
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=30)
fit = bestfeatures.fit(Xx,Yy)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(Xx.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(30,'Score'))  #print 10 best features

###################important feature selection 6th techniques
# Select features using chi-squared test
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

## Get score using original model
logreg = LogisticRegression(C=1)
logreg.fit(x_train, y_train)
scores = cross_val_score(logreg, x_train, y_train, cv=10)
print('CV accuracy (original): %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
highest_score = np.mean(scores)

## Get score using models with feature selection
for i in range(1, x_train.shape[1]+1, 1):
    # Select i features
    select = SelectKBest(score_func=chi2, k=i)
    select.fit(x_train, y_train)
    x_train_selected = select.transform(x_train)

    # Model with i features selected
    logreg.fit(x_train_selected, y_train)
    scores = cross_val_score(logreg, x_train_selected, y_train, cv=10)
    print('CV accuracy (number of features = %i): %.3f +/- %.3f' % (i, 
                                                                     np.mean(scores), 
                                                                     np.std(scores)))
    
    # Save results if best score
    if np.mean(scores) > highest_score:
        highest_score = np.mean(scores)
        std = np.std(scores)
        k_features_highest_score = i
    elif np.mean(scores) == highest_score:
        if np.std(scores) < std:
            highest_score = np.mean(scores)
            std = np.std(scores)
            k_features_highest_score = i
        
# Print the number of features
print('Number of features when highest score: %i' % k_features_highest_score)

# Select features
select = SelectKBest(score_func=chi2, k=k_features_highest_score)
select.fit(x_train, y_train)
X_train_selected = select.transform(x_train)
# Fit model
logreg = LogisticRegression(C=1)
logreg.fit(X_train_selected, y_train)
# Model performance with selected features
scores = cross_val_score(logreg, X_train_selected, y_train, cv=10)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

#%%
# #################Printing the decisin tree################
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(x_train, y_train)
clf.plot_tree(clf)
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier(random_state = 0)
dtree.fit(x_train, y_train)

#%%
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

#%%[markdown]
########################  K-Best Selector ####################
## Running the model with the selected important features

# Select features using chi-squared test
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#%%
# Select features
select = SelectKBest(score_func=chi2, k=20)
select.fit(x_train, y_train)
X_train_selected = select.transform(x_train)
X_test_selected = select.transform(x_test)
#%%
rf2 = RandomForestClassifier(n_estimators = 10, random_state = 0)
rf2.fit(X_train_selected, y_train)
print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(rf2.score(X_test_selected,y_test)*100))
####################Other method
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# Select features
select = SelectKBest(score_func=chi2, k=20)
select.fit(x_tran_abs, y_train)
X_train_selected = select.transform(x_tran_abs)

select2=SelectKBest(score_func=chi2, k=20)
select2.fit(x_test_abs, y_test)
X_test_selected = select2.transform(x_test_abs)

############################Another method to get Highest predictors##########
# Select features using chi-squared test
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

## Get score using original model
logreg = LogisticRegression(C=1)
logreg.fit(x_train, y_train)
scores = cross_val_score(logreg, x_train, y_train, cv=10)
print('CV accuracy (original): %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
highest_score = np.mean(scores)

## Get score using models with feature selection
for i in range(1, x_train.shape[1]+1, 1):
    # Select i features
    select = SelectKBest(score_func=chi2, k=i)
    select.fit(x_train, y_train)
    x_train_selected = select.transform(x_train)

    # Model with i features selected
    logreg.fit(x_train_selected, y_train)
    scores = cross_val_score(logreg, x_train_selected, y_train, cv=10)
    print('CV accuracy (number of features = %i): %.3f +/- %.3f' % (i, 
                                                                     np.mean(scores), 
                                                                     np.std(scores)))
    
    # Save results if best score
    if np.mean(scores) > highest_score:
        highest_score = np.mean(scores)
        std = np.std(scores)
        k_features_highest_score = i
    elif np.mean(scores) == highest_score:
        if np.std(scores) < std:
            highest_score = np.mean(scores)
            std = np.std(scores)
            k_features_highest_score = i
        
# Print the number of features
print('Number of features when highest score: %i' % k_features_highest_score)

####prediction with highest predictors
# Select features
select = SelectKBest(score_func=chi2, k=k_features_highest_score)
select.fit(x_train, y_train)
X_train_selected = select.transform(x_train)
# Fitting model with those important features only
# Fit model
logreg = LogisticRegression(C=1)
logreg.fit(X_train_selected, y_train)
# Accessing the model performance
# Model performance with selected features
scores = cross_val_score(logreg, X_train_selected, y_train, cv=10)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

#%%[markdown]
###################### Prediction on Validation set########################
valData.dtypes

#%%
# Convert ID to integer inorder to skip from dummies
valData['StudentID']=valData['StudentID'].astype(int)

#%%
# get dummies for categorical and object
valdata_dummy = pd.get_dummies(valData, drop_first=True) 

#%%
valdata_dummy.shape, data.shape

#%%
# Segredation of explanatory and Target variables
valx_data = valdata_dummy[valdata_dummy.loc[:, valdata_dummy.columns != 'Target'].columns]
valy = valdata_dummy['Target']

#%%
# DropID for modeling
valx_data.drop(['StudentID'], axis=1, inplace=True)
#%%
# Modeling
new_lr = LogisticRegression(C=0.03);
new_lr.fit(xx_train,y_train);
print("Test Accuracy {:.2f}%".format(new_lr.score(xx_test,y_test)*100));

#%%
from sklearn.ensemble import RandomForestClassifier
new_rf = RandomForestClassifier(n_estimators = 10, random_state = 2)
new_rf.fit(xx_train,y_train)
print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(new_rf.score(xx_test,y_test)*100))

#%%
#Prediction on validation set ( have to follow all the procedure of cleaning and dummies all as in train and 
# test data)
valdata_pred_lr=new_rf.predict(valx_data)
#%%
#Create a  DataFrame with the Member ids and our prediction regarding whether they disenrolled or not
IDwitPrediction_lr=pd.DataFrame({'MemberId':valdata_dummy['MemberID'],'VolDisenrollPrediction':valdata_pred_lr})
#%%
IDwitPrediction_lr.head()
#IDwitPrediction_lr.MemberId.astype(str)
#IDwitPrediction_lr.VolDisenrollPrediction.astype(str)

#%%
# Export the predicted values in excel sheet
IDwitPrediction_lr.to_excel('MA_Disenrollment_3.xlsx',index=False)


#%%
# Firstly create a table in the sql server as 
#DROP TABLE EDWBronze.Schema.CP_School_Predicted_2nd;
#CREATE TABLE EDWBronze.Schema.CP_School_Predicted_2nd([StudentID] INT,[Prediction] INT)
# Exporting the data to SQL server
# connecting python to the SQL server to push data to SQL Server
server= 'prd_75.sql.caresource.corp\prd_75'
db='EDWBronze'

connStr = pyodbc.connect('DRIVER={SQL Server}; SERVER=' +server+ '; DATABASE='+db+ '; Trusted_Connection=yes')
cursor = connStr.cursor()

for index,row in IDwitPrediction_lr.iterrows():
    cursor.execute("INSERT INTO EDWBronze.Schema.CP_School_Predicted_2nd([StudentID],[Prediction]) values (?, ?)", row['MemberId'], row['VolDisenrollPrediction']) 
    connStr.commit()
cursor.close()
connStr.close()


#%%
####################Plotting the Learning Curve and validation##################################
#============================Learning curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
# plotting the learning curve
# Define a fucntion which checks for the learning curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Validation score")

    plt.legend(loc="best")
    return plt


# Plot learning curves
title = "Learning Curves (Logistic Regression)"
cv = 20
plot_learning_curve(lr, title, x_train, y_train, ylim=(0.3, 1.01), cv=cv, n_jobs=1);
#%%[markdown]
##Learning curves in a nutshell:

#Learning curves allow us to diagnose if the is overfitting or underfitting. When the model 
# overfits, it means that it performs well on the training set, but not not on the 
# validation set. Accordingly, the model is not able to generalize to unseen data. 
# If the model is overfitting, the learning curve will present a gap between the training 
# and validation scores. Two common solutions for overfitting are reducing the complexity 
# of the model and/or collect more data. On the other hand, underfitting means that the 
# model is not able to perform well in either training or validations sets. In those cases, 
# the learning curves will converge to a low score value. When the model underfits, 
# gathering more data is not helpful because the model is already not being able to 
# learn the training data. Therefore, the best approaches for these cases are to improve 
# the model (e.g., tuning the hyperparameters) or to improve the quality of the 
# data (e.g., collecting a different set of features). Discussion of our results:

#The model doesn't overfit. As we can see, the curves converge and no gap between the 
# training and the validation score exists in the last points of the curve. The model
#  underfits. Our final score is about 0.786. Although our model does better predictions 
# than a flip-a-coin strategy, it is still far from being an 'intelligent' model. 
# For now, it's just an 'artificial' model.

#%%
#==========================Validation curve=========================
# Plot validation curve
def plot_validation_curve(estimator, title, X, y, param_name, param_range, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    train_scores, test_scores = validation_curve(estimator, X, y, param_name, param_range, cv)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(param_range, train_mean, color='r', marker='o', markersize=5, label='Training score')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='r')
    plt.plot(param_range, test_mean, color='g', linestyle='--', marker='s', markersize=5, label='Validation score')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='g')
    plt.grid() 
    plt.xscale('log')
    plt.legend(loc='best') 
    plt.xlabel('Parameter') 
    plt.ylabel('Score') 
    plt.ylim(ylim)
# apply the function
# Plot validation curve
title = 'Validation Curve (Logistic Regression)'
param_name = 'C'
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0] 
cv = 10
plot_validation_curve(estimator=logreg, title=title, X=x_train, y=y_train, param_name=param_name,ylim=(0.5, 1.01), 
                      param_range=param_range);
#%%[markdown]
#Validation curves in a nutshell:

#Validation curves are a tool that we can use to improve the performance of our model. 
# It counts as a way of tuning our hyperparameters. They are different from the learning 
# curves. Here, the goal is to see how the model parameter impacts the training and validation 
# scores. This allow us to choose a different value for the parameter, to improve the
#  model. Once again, if there is a gap between the training and the validation score, 
# the model is probably overfitting. In contrast, if there is no gap but the score value 
# is low, we can say that the model underfits. Discussion of our results:

#The figure shows that there is no huge difference in model's performance as far as 
# we choose a C value of 10−1 or higher. Note that in a logistic regression, C is the 
# only model parameter that we can change

#%%
##########################Validation Curve ########################################



#%%
########################## Grid search for parameter tuning #######################
# Obtain the best parameter
rfc=RandomForestClassifier(random_state=42)
param_grid = { 'n_estimators': [200, 500], 'max_features': ['auto', 'sqrt', 'log2'],
               'max_depth' : [4,5,6,7,8], 'criterion' :['gini', 'entropy'] }

# Fit the grid search
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(x_train, y_train)

# get the best parameter
CV_rfc.best_params_
# above value gave: following parameters
# max_feature='auto', n_estimators=200, max_depth=8, criterion='gini'
# fitting mo
rfc1=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 200, max_depth=8, criterion='gini')
rfc1.fit(x_train, y_train)
# predict and get accuracy
pred=rfc1.predict(x_test)
print("Accuracy for Random Forest on CV data: ",accuracy_score(y_test,pred))

# Other method
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

clf = RandomForestClassifier()

parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }


acc_scorer = make_scorer(accuracy_score)

grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

clf = grid_obj.best_estimator_

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))

#######Other method of grid search
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
class RandomForestClassifierWithCoef(RandomForestClassifier):
    def fit(self, *args, **kwargs):
        super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_
# apply the class define by converting to object
from sklearn.feature_selection import RFECV
rf = RandomForestClassifierWithCoef()
rfecv = RFECV(estimator=rf, step=1, cv=3, scoring='accuracy')
rfecv.fit(X_train, y_train)

print("Optimal number of features : %d" % rfecv.n_features_)


#%%
#################### K-fold Cross validation ##############################
from sklearn.cross_validation import KFold

kf = KFold(891, n_folds=10)    
outcomes = []
    
fold = 0
for train_index, test_index in kf:
    fold += 1
    X_train, X_test = X_all.values[train_index], X_all.values[test_index]
    y_train, y_test = y_all.values[train_index], y_all.values[test_index]
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    outcomes.append(accuracy)
    print("Fold {0} accuracy: {1}".format(fold, accuracy))     
mean_outcome = np.mean(outcomes)
print("\n\nMean Accuracy: {0}".format(mean_outcome)) 

#%%


########################COmpletely other techniques####################
# Here we devide into continous and categorical: continous data
#  transformation and categorical to label encoding
# segregation of continous and categorical data
df_cate=df1.select_dtypes(include=['object', 'category'])
df_cont=df1.select_dtypes(exclude=['object','category'])
# drop the target variable
ContDf=df_cont.drop('Voluntary_disenroll', axis=1)

#%%
#summary statistics for continous variable
ContDf.describe()

#%%
ContDf.columns
#%%
# Continous features without target
confeatures_notarget = [col for col in df_cont.columns if col!='Voluntary_disenroll']

#%%
# droping all the continous features
dropcol=list(confeatures_notarget)
CateDatawithTarget=df1.drop(dropcol, axis=1)

#%%
# cross verifying the categorical data including target
CateDatawithTarget.shape

#%%
# Histogram for all continuous variables
ContDf.hist(figsize=(20,30), bins=10, xlabelsize=3, ylabelsize=4);

#%%
# Pair plot to 
#sns.pairplot(ContDf)

#%%
# other method to get categorical with target
to_drop=ContDf.columns
df_catewithTarget=df[df.columns.difference(to_drop)]
df_catewithTarget.shape

#%%
CateDatawithTarget.head()

#select categorical features without target variable
cate_featurenoTarget=[col for col in CateDatawithTarget
                      .columns if col!='target']

df_cate_removed['MemberID'] = df_cate_removed['MemberID'].astype(int)

#One-hot encoding for the categorical features
df_cate_encoded=pd.get_dummies(df_cate_removed, drop_first=True)  
# Shape of data after encoded
df_cate_encoded.shape
# Lets drop target variable
df_cont_filledNotarget=df_cont_filled.drop('Voluntary_disenroll', axis=1)
# standard scaling data
from sklearn import preprocessing
names = df_cont_filledNotarget.columns
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df_cont_filledNotarget)
scaled_df = pd.DataFrame(scaled_df, columns=names)
scaled_df.head()
X1=pd.concat((scaled_df,df_cont_filled['Voluntary_disenroll'] ),axis=1)
#Combining all the data back to one table
X=pd.concat((df_cate_encoded, X1),axis=1)
# Target and explanatory variable 
x_data = X[X.loc[:, X.columns != 'Voluntary_disenroll'].columns]
y = X['Target']
# Train and Test split of data 80-20 
x_train, x_test, y_train, y_test = train_test_split(x_data,y,test_size = 0.2,random_state=0)


#%%

#######################################NLP #####################################
##############LDA Topic Modeling: Theme understanding ########################
################################################################################

# Just trying to hide the printing warning or status updata
import warnings; warnings.filterwarnings('ignore')
#%%
# importing the required packages
import numpy as np
import pandas as pd
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline
import scipy
import sys
import csv

# Importing the additional packages: plotly packages for the visualization

import plotly.offline as pyo
pyo.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

#%%
# importing Natural language Toolkit (nltk) packages
import nltk
# importing Regextocknizeer, corpora, models and all
from nltk.tokenize import RegexpTokenizer
#from stop_words import get_stop_words
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora, models
import gensim
#import pyLDAvis.gensim
# import the stop-words for english
from nltk.corpus import stopwords
nltk.download('stopwords');
stopwords = stopwords.words('english');

#%%
# verifying the stopwords
stopwords

#%%
# Other required packages from Scikit-learn for vectorization, feature extraction, LDA
from collections import Counter
from scipy.misc import imread
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


#%%
# reading the Input from csv
df = pd.read_csv('MA_Disenroll_CALL_TYPES_05319.csv' ,header=0,encoding = 'unicode_escape')

#%%
df.head()

#%%
df.shape

#%%
# Visualizing the sample text data for few examples
print(df['CSCF_TEXT'].head())

#%%
# Checking the different categories 
df['CATEGORY'].unique()

#%%
# Creating the new dataset with following three things
review_df=df[['CSSC_ID','CSCF_TEXT', 'CATEGORY']]

#%%
# looking the dataset
review_df.head()

#%%
# Incase of need: Storing in the new datafram
review_df_raw=review_df.copy()

#%%
# replacing na values in category with No category 
review_df["CATEGORY"].fillna("NoCategory", inplace = True);

#%%
# validating our imputation
review_df.head()

#%%
# checking all the distinct values in the new data set
review_df.CATEGORY.unique()

#%% [markdown]
# ## Summary statistics
# - in this section I visualize the basic statistics like distribution of entries for each category of the call.
# - I plotted the simple bar plot to visualize it

#%%
# importing the required plotly for visulization
import plotly
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
pyo.init_notebook_mode(connected=True)
plotly.offline.init_notebook_mode(connected=True)

#%%
# Visualization of words per categories before cleaning

z = {'MEDICAL': 'Medical', 'PHARMACY': 'Pharmacy', 'DENTAL': 'Dental', 'VISION': 'Vision', 'BEHAVIORAL HEALTH' : 'Bev_health', 'NoCategory':'Others'}
data = [go.Bar(
    x = review_df.CATEGORY.map(z).unique(),
            y = review_df.CATEGORY.value_counts().values,
            marker= dict(colorscale='Jet',color = review_df.CATEGORY.value_counts().values),
            text='Documents distribution (text entries) to Category')]

layout = go.Layout(autosize=False,width=500,
    height=500,title='Target variable distribution')

fig = go.Figure(data=data, layout=layout)

plotly.offline.iplot(fig, filename='bar-direct-labels')

#%% [markdown]
## Word frequency plot to get odd  
# - Here I will focus anything odd about the odd appearance in the word frequency plot
# - notice, these words actually do tell some thing or don't tell much about the theme and concepts that I want to portray 

#%%
# Word frequencies visualization in the dataset before cleaning
all_words = review_df['CSCF_TEXT'].str.split(expand=True).unstack().value_counts()
data = [go.Bar(
            x = all_words.index.values[2:50],
            y = all_words.values[2:50],
            marker= dict(colorscale='Jet',
                         color = all_words.values[2:100]
                        ),
            text='Word counts'
    )]

layout = go.Layout(autosize = False, width=1000, height=1000,
    title='Top 50  Word frequencies in the dataset (but it is Uncleaned)'
)

fig = go.Figure(data=data, layout=layout)

pyo.iplot(fig, filename='basic-bar')

#%% [markdown]
#- Store the text of each categories in a Python list

#   - We first create six different python lists that store the texts of 'MEDICAL', 'NoCategory', 'PHARMACY', 'VISION', 'DENTAL',
 #      'BEHAVIORAL HEALTH' respectively as follows:


#%%
# category values
review_df.CATEGORY.unique()


#%%
# Storing the documents related to each categories in the corresponding array
medical = review_df[review_df.CATEGORY=="MEDICAL"]["CSCF_TEXT"].values
pharmacy = review_df[review_df.CATEGORY=="PHARMACY"]["CSCF_TEXT"].values
vision = review_df[review_df.CATEGORY=="VISION"]["CSCF_TEXT"].values
dental = review_df[review_df.CATEGORY=="DENTAL"]["CSCF_TEXT"].values
behav_health = review_df[review_df.CATEGORY=="BEHAVIORAL HEALTH"]["CSCF_TEXT"].values
other = review_df[review_df.CATEGORY=="NoCategory"]["CSCF_TEXT"].values

#%%
type(medical)

#%% [markdown]
# In the Dataframe type


#%%
# filtering dataframe for Medical, Pharmacy, Vision category in a dataframe
df_medical=review_df.loc[review_df['CATEGORY'] == "MEDICAL"]
df_pharmacy=review_df.loc[review_df['CATEGORY'] == "PHARMACY"]
df_vision=review_df.loc[review_df['CATEGORY'] == "VISION"]

#%% [markdown]
# Now the time of showing wordcloud for each categories


#%%
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

#%%
type(medical_text)
#%%
# couting total words on medical
# for medical
medical_text = " ".join(text for text in medical)
print ("There are {} words in the combination of all text in medical category.".format(len(medical_text)))

#%%
# Generate a word cloud image
plt.figure(figsize=(12,10))
wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", max_words=10000, max_font_size= 30)
wordcloud.generate(" ".join(medical))
# display my wordcloud
plt.title("Popular words in Medical Category", fontsize=20)
plt.imshow(wordcloud, interpolation='bilinear')
#plt.imshow(wordcloud.recolor( colormap= 'Pastel2' , random_state=17), alpha=0.98)
plt.axis("off")
plt.show()

#%%
# Text processing
# first create the stopwords and then update with you words to be removed
from nltk.corpus import stopwords
stop_words1 = set(stopwords.words("english"))
my_words_to_removed=['member' , 'advised','ohio', 'dayton', 'cc','ca','oh','md','kpn','blvd','james','inc', 
                     'ste','b', 'oaks','c', 'ccs','us','th','eir', 'rd', 'llc','bbw','phone','number',
                     'avon', 'smith', 'detroit','pho', 'ty', 'w','st','fl','cl','ne','l','hy','st', 'n', 
                     'ave','cs','g','dos','par','p','lan','zhang', 'cleveland','ralph', 'mh', 'jr','r',
                     'amy', 'eob', 'rn','e','rn','called','call', 'caller','must','get','state','caresource','day','name','see','dr','id','use','due','would','state']
new_stopwords_list = stop_words1.union(my_words_to_removed)

#%%
# define  tokenizer adn lemmatizer
pattern = r'\b[^\d\W]+\b'
tokenizer = RegexpTokenizer(pattern)
lemmatizer = WordNetLemmatizer()
#%%
# pre-processing 
import re
import string
#converting into lower-case
medical_text=medical_text.lower()
#removing the numbers
after_number = re.sub(r'\d+', '', medical_text)
#Removing punctuation
after_punctuation = after_number.translate(str.maketrans('','', string.punctuation))
#removing with space if they have
after_whitespace = after_punctuation.strip()
# Tokenization of the words
tokens_words=tokenizer.tokenize(after_whitespace)
# removing stop words
no_stop_words=[raw for raw in tokens_words if not raw in new_stopwords_list]
# lemmatization
lemmatization_words = [lemmatizer.lemmatize(tokens_words) for tokens_words in no_stop_words]
# remove word containing only single char
new_lemma_tokens = [raw for raw in lemmatization_words if not len(raw) == 1]

#%%
medicaltext=new_lemma_tokens
medicaltext

#%% [markdown]
## Visualizing the top words in medical category after cleaning

#%% 
type(medicaltext)

#%%
# Word frequencies visualization in the dataset after cleaning
from collections import Counter
word_list1 = [item for item in medicaltext]
#word_list = texts


counts = dict(Counter(word_list1).most_common(50))

labels, values = zip(*counts.items())

# sort your values in descending order
indSort = np.argsort(values)[::-1]

# rearrange your data
labels = np.array(labels)[indSort]
values = np.array(values)[indSort]

indexes = np.arange(len(labels))
#all_words = review_df['CSCF_TEXT'].str.split(expand=True).unstack().value_counts()
data = [go.Bar(
            x = labels,
            y = values,
            marker= dict(colorscale='Jet',
                         color = all_words.values[2:100]
                        ),
            text='Word counts'
    )]

layout = go.Layout(autosize=False, width=1200, height=500,
    title='Top 50  Word frequencies in the dataset (Cleaned)'
)

fig = go.Figure(data=data, layout=layout)

pyo.iplot(fig, filename='basic-bar')

#%%
# split() returns list of all the words in the string 
from collections import Counter
#split_it = medical_text.split() 

# Pass the split_it list to instance of Counter class. 
Counter1 = Counter(medicaltext) 
  
# most_common() produces k frequently encountered 
# input values and their respective counts. 
most_occur1 = Counter1.most_common(15) 

#%% [markdown]
# Now count tuple values of top words
most_occur1

#%% 
# counting words after cleaning
medical_text1 = " ".join(text for text in medicaltext)
print ("There are {} words in the combination of all text in medical category.".format(len(medical_text1)))

#%% [markdown]
## Now Topic modeling for Medical text data 
# -(one-gram but need to have bi-gram Topic modeling (further action needed))


#%%
# defining the specific words that to be removed, give punctuation and instenciate lemmatizer
#from nltk.corpus import stopwords
pattern = r'\b[^\d\W]+\b'
tokenizer = RegexpTokenizer(pattern)
lemmatizer = WordNetLemmatizer()
remove_words1=['id' , , 'cc','ca','oh','md','kpn','blvd','james','inc', 'ste','b', 'oaks','c', 'ccs','us','th','eir',
                     'rd', 'llc','bbw','phone','number','avon', 'smith', 'detroit','pho', 'ty', 'w','st','fl','cl','ne','l','hy','st', 'n', 'ave',
                    'cs','g','dos','par','p','lan','zhang', 'cleveland','ralph', 'mh', 'jr','r','amy', 'eob', 'rn','e','rn','advised','called',
                    'state','cincinnati','county','call', 'caller','use','office','day','would','must','id','dr','file', 'st','much', 'state',
                    'within', 'per','must','due','told','want','see','columbus', 'sent','id','contact','date','may','team','contacted','kit','top' ]

#%%
from nltk.corpus import stopwords
nltk.download('stopwords');
stopwords = stopwords.words('english');
#%%
# list for tokenized documents in loop
text_medical = []

# loop through document list of the column in CSCF_TEXT
for i in df_medical['CSCF_TEXT'].iteritems():
    # clean and tokenize document string
    raw = str(i[1]).lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [raw for raw in tokens if not raw in stopwords]
    
    # remove stop words from tokens
    stopped_tokens_new = [raw for raw in stopped_tokens if not raw in remove_words1]
    
    # lemmatize tokens
    lemma_tokens = [lemmatizer.lemmatize(tokens) for tokens in stopped_tokens_new]
    
    # remove word containing only single char
    new_lemma_tokens = [raw for raw in lemma_tokens if not len(raw) == 1]
    
    # add tokens to list
    text_medical.append(new_lemma_tokens)

# sample data
print(text_medical[0])

#%% [markdown]
## Create the dictionary
# assign unique id to the unique token: create a dictionary
#%%
dictionary1 = corpora.Dictionary(text_medical)


#%% [markdown]
## Creating the term -document matrix



#%%
# convert dictionary into bag of words
corpus1 = [dictionary1.doc2bow(textes) for textes in text_medical]

#%%
#Lets check the corpus 
print(corpus1[0])

#%%
# let's check the term id for each
print(dictionary1.token2id)

#%% [markdown]
## Creating the LDA Model
#%%
# we can put any number of topic: here I am using only 6 topics, bigger the passes no more accurate model

ldamodel1 = gensim.models.ldamodel.LdaModel(corpus1, num_topics=6, id2word = dictionary1, passes=20);


#%%
# printing out the result
ldamodel1.print_topics(num_topics=4, num_words=8)
#print(ldamodel.print_topics(num_topics=6, num_words=3))

#%%
# LDA printing
import pprint
pprint.pprint(ldamodel1.top_topics(corpus1,topn=5));

#%% [markdown]
## LDA Visualization

#%%
# importing LDA visualizing packages
import pyLDAvis as ps
from pyLDAvis import enable_notebook, gensim 
#%%
#LDA visualization
ps.enable_notebook()
ps.gensim.prepare(ldamodel1, corpus1, dictionary1)

#%% [markdown]
## 
#The Bussiness need is to identify the root cause for ..

#Here in this project, I am trying to do the basic topic modeling for the .. dataset. 
# Here by Topic modeling I means to say, I am trying to uncover the abstract theme or topic 
# based on the underlying words and sentences in a corpus of text. I will incoperate following techniques: 
# - LDA (Latent Dirichelet Allocation) 

#I will also perform some basics of NLP such as 
# - Tokenization
# - Lemmatization
# - Vectorization

# The General outline of the project
# - EDA (Exploratory Data Analysis) and Wordclouds
# - Analyzing the data by generating simple 
# statistics such word frequencies over the different areas as well as plotting some wordclouds.

# - NLP (Natural Language Processing) with NLTK (Natural Language Toolkit)
# - Introducing basic text processing methods such as tokenizations, stop word removal, 
# stemming andvectorizing text via term frequencies (TF) as well as the inverse document frequencies (TF-IDF)

# - Topic Modelling with LDA and NNMF 
# - Implementing the two topic modelling techniques of Latent Dirichlet Allocation (LDA) 
# and Non-negativeMatrix Factorization (NMF).

######################LDA
#%%
# Just trying to hide the printing warning or status updata
import warnings; warnings.filterwarnings('ignore')
# Importing the general packages
import numpy as np
import pandas as pd
import base64    #for encoding and decoding data
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import csv
import sys
import re
import scipy
# Importing the additional packages: plotly packages for the visualization
import plotly.offline as pyo
pyo.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

# importing Natural language Toolkit (nltk) packages
import nltk
# importing Regextocknizeer, corpora, models and all
from nltk.tokenize import RegexpTokenizer
#from stop_words import get_stop_words
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora, models
import gensim
#import pyLDAvis.gensim
# import the stop-words for english
from nltk.corpus import stopwords
nltk.download('stopwords');
stopwords = stopwords.words('english');
# Check out the stopwords for the varification
stopwords

# Other required packages from Scikit-learn for vectorization, feature extraction, LDA
from collections import Counter
from scipy.misc import imread
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from matplotlib import pyplot as plt
# reading the Input from csv
df = pd.read_csv('data.csv',header=0,encoding = 'unicode_escape')
# Visualizing the sample text data for few examples
print(df['CSCF_TEXT'].head())
# Checking the different categories 
df['CATEGORY'].unique()
# Creating the new dataset with following three things
review_df=df[['studentID','CSCF_TEXT', 'CATEGORY']]
# looking the dataset
review_df.head()
# Incase of need
review_df_raw=review_df.copy()
# replacing na values in category with No category 
review_df["CATEGORY"].fillna("NoCategory", inplace = True);
# validating our imputation
review_df.head()
# checking all the distinct values in the new data set
review_df.CATEGORY.unique()
#%%[markdown]
# Summary statistics
#- in this section I visualize the basic statistics like distribution of entries for each 
# category of the call.
#- I plotted the simple bar plot to visualize it
#%%
# importing the required plotly for visulization
import plotly
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
pyo.init_notebook_mode(connected=True)
plotly.offline.init_notebook_mode(connected=True)
# Visualization of words per categories before cleaning
z = {'MEDICAL': 'Medical', 'PHARMACY': 'Pharmacy', 'DENTAL': 'Dental', 'VISION': 'Vision', 'BEHAVIORAL HEALTH' : 'Bev_health', 
     'NoCategory':'Others'}
data = [go.Bar(x = review_df.CATEGORY.map(z).unique(),
            y = review_df.CATEGORY.value_counts().values,
            marker= dict(colorscale='Jet',color = review_df.CATEGORY.value_counts().values),
            text='Text entries attributed to Category')]

layout = go.Layout(title='Target variable distribution')

fig = go.Figure(data=data, layout=layout)

plotly.offline.iplot(fig, filename='bar-direct-labels')

#%%
# Word frequency plot to get odd  
#- Here I will focus anything odd about the odd appearance in the word frequency plot
#- notice, these words actually do tell some thing or don't tell much about the theme 
# and concepts that I want to portray 
#%%
# Word frequencies visualization in the dataset before cleaning
all_words = review_df['CSCF_TEXT'].str.split(expand=True).unstack().value_counts()
data = [go.Bar(
            x = all_words.index.values[2:50],
            y = all_words.values[2:50],
            marker= dict(colorscale='Jet',
                         color = all_words.values[2:100]
                        ),
            text='Word counts'
    )]

layout = go.Layout(width=1000, height=500,
    title='Top 50  Word frequencies in the dataset (but it is Uncleaned)'
)

fig = go.Figure(data=data, layout=layout)

pyo.iplot(fig, filename='basic-bar')
# category values
review_df.CATEGORY.unique()
medical = review_df[review_df.CATEGORY=="MEDICAL"]["CSCF_TEXT"].values
pharmacy = review_df[review_df.CATEGORY=="PHARMACY"]["CSCF_TEXT"].values
vision = review_df[review_df.CATEGORY=="VISION"]["CSCF_TEXT"].values
dental = review_df[review_df.CATEGORY=="DENTAL"]["CSCF_TEXT"].values
behav_health = review_df[review_df.CATEGORY=="BEHAVIORAL HEALTH"]["CSCF_TEXT"].values
other = review_df[review_df.CATEGORY=="NoCategory"]["CSCF_TEXT"].values
#Taking only medical
df_medical=review_df.loc[review_df['CATEGORY'] == "MEDICAL"]
df_pharmacy=review_df.loc[review_df['CATEGORY'] == "PHARMACY"]
df_vision=review_df.loc[review_df['CATEGORY'] == "VISION"]
# WORDCLOUD
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# if you want to updata the stopwords by adding some other words do as follows

#stopwords=set(STOPWORDS)
#stopwords.updata(["Chandi","Ohio","Dayton"])
# Medical category
# couting total words on medical
# for medical
medical_text = " ".join(text for text in medical)
print ("There are {} words in the combination of all text in medical category."
        .format(len(medical_text)))

# Generate a word cloud image
plt.figure(figsize=(16,13))
wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", max_words=10000, max_font_size= 30)
wordcloud.generate(" ".join(medical))
# display my wordcloud
plt.title("Popular words in Medical Category", fontsize=20)
plt.imshow(wordcloud, interpolation='bilinear')
#plt.imshow(wordcloud.recolor( colormap= 'Pastel2' , random_state=17), alpha=0.98)
plt.axis("off")
plt.show()

#Pre-processing data from medical_text
#define stop words including your words to be removed
# first create the stopwords and then update with you words to be removed
from nltk.corpus import stopwords
stop_words1 = set(stopwords.words("english"))
my_words_to_removed=['member' , 'ohio', 'dayton', 'cc','ca','oh','md','kpn','blvd','james','inc', 'ste','b', 'oaks','c', 'ccs','us','th','eir',
                     'rd', 'llc','bbw','phone','number','avon', 'smith', 'detroit','pho', 'ty', 'w','st','fl','cl','ne','l','hy','st', 'n', 
                     'ave',
                    'cs','g','dos','par','p','lan','zhang', 'cleveland','ralph', 'mh', 'jr','r','amy', 'eob', 'rn','e','rn']
new_stopwords_list = stop_words1.union(my_words_to_removed)

# pre-processing 
import re
import string
#converting into lower-case
medical_text=medical_text.lower()
#removing the numbers
after_number = re.sub(r'\d+', '', medical_text)
#Removing punctuation
after_punctuation = after_number.translate(str.maketrans('','', string.punctuation))
#removing with space if they have
after_whitespace = after_punctuation.strip()
# Tokenization of the words
tokens_words=tokenizer.tokenize(after_whitespace)
# removing stop words
no_stop_words=[raw for raw in tokens_words if not raw in new_stopwords_list]
# lemmatization
lemmatization_words = [lemmatizer.lemmatize(tokens_words) for tokens_words in no_stop_words]
# remove word containing only single char
new_lemma_tokens = [raw for raw in lemmatization_words if not len(raw) == 1]

new_lemma_tokens
medicaltext=new_lemma_tokens
# split() returns list of all the words in the string 
from collections import Counter
#split_it = medical_text.split() 

# Pass the split_it list to instance of Counter class. 
Counter1 = Counter(medicaltext) 
  
# most_common() produces k frequently encountered 
# input values and their respective counts. 
most_occur1 = Counter1.most_common(15) 
most_occur1
#%%
# Topic modeling for medical text
# defining the specific words that to be removed, give punctuation and instenciate lemmatizer
#from nltk.corpus import stopwords
pattern = r'\b[^\d\W]+\b'
tokenizer = RegexpTokenizer(pattern)
lemmatizer = WordNetLemmatizer()
remove_words1=['member' , 'ohio', 'dayton', 'cc','ca','oh','md','kpn','blvd','james','inc', 'ste','b', 'oaks','c', 'ccs','us','th','eir',
                     'rd', 'llc','bbw','phone','number','avon', 'smith', 'detroit','pho', 'ty', 'w','st','fl','cl','ne','l','hy','st', 'n', 'ave',
                    'cs','g','dos','par','p','lan','zhang', 'cleveland','ralph', 'mh', 'jr','r','amy', 'eob', 'rn','e','rn','advised','called']

# list for tokenized documents in loop
text_medical = []
#%%
# loop through document list of the column in CSCF_TEXT
for i in df_medical['CSCF_TEXT'].iteritems():
    # clean and tokenize document string
    raw = str(i[1]).lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [raw for raw in tokens if not raw in stopwords]
    
    # remove stop words from tokens
    stopped_tokens_new = [raw for raw in stopped_tokens if not raw in remove_words1]
    
    # lemmatize tokens
    lemma_tokens = [lemmatizer.lemmatize(tokens) for tokens in stopped_tokens_new]
    
    # remove word containing only single char
    new_lemma_tokens = [raw for raw in lemma_tokens if not len(raw) == 1]
    
    # add tokens to list
    text_medical.append(new_lemma_tokens)

# sample data
print(text_medical[0])
# assign unique id to the unique token: create a dictionary
dictionary1 = corpora.Dictionary(text_medical)
# convert dictionary into bag of words
corpus1 = [dictionary1.doc2bow(textes) for textes in text_medical]
#Lets check the corpus 
print(corpus1[0])
# let's check the term id for each
print(dictionary1.token2id)
# we can put any number of topic: here I am using only 6 topics, bigger the passes no more accurate model
ldamodel1 = gensim.models.ldamodel.LdaModel(corpus1, num_topics=6, id2word = dictionary1, passes=20);
# printing out the result
ldamodel1.print_topics(num_topics=6, num_words=8)
#print(ldamodel.print_topics(num_topics=6, num_words=3))
# printing out the result
ldamodel1.print_topics(num_topics=6, num_words=8)
#print(ldamodel.print_topics(num_topics=6, num_words=3))
#LDA visualization
ps.enable_notebook()
ps.gensim.prepare(ldamodel1, corpus1, dictionary1)

#%%
#Topic modeling for all data
#Here we are going to do few pro-processing steps: 
# - pre-processing is required inorder to convert our raw data into the readle format for the machine and our model. 

#These are the few pre-processing steps I will go over:
# - Tokenization 
#    - it is the process of segregating of the text into its individual constitutent words.
#- Stopwords 
#  - Throw away any words that occur too frequently and unfrequently (common words they don't give good meaning)
#- Stemming 
#  - identify the parent word that convey the same meaning with other words and combine 
# different words into a single parent word
#- Vectorization 
# - we convert text into vector format from differnt following method
# - bag-of-words (we call vectorization of the raw text)
# - where we create a matrix for each text or documents in the corpus
# - this matrix stores word frequencies (word count) 
#%%
# visualizing the text data for each documents: sample data
print(df['CSCF_TEXT'].head(20))
#%%
# Performing text-mining: data pre-processing
#- Perform Tokenization, Words removal, and Lemmatization
# defining the specific words that to be removed, give punctuation and instenciate lemmatizer
pattern = r'\b[^\d\W]+\b'
tokenizer = RegexpTokenizer(pattern)
lemmatizer = WordNetLemmatizer()
remove_words=['Dayton', 'Ohio', 'ca', 'ccs', 'md', 'oh', 'james', 
              'member','day', 'phone number', 'number','advised']
#%%
# # list for tokenized documents in loop
texts = []

# loop through document list of the column in CSCF_TEXT
for i in df['CSCF_TEXT'].iteritems():
    # clean and tokenize document string
    raw = str(i[1]).lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [raw for raw in tokens if not raw in stopwords]
    
    # remove stop words from tokens
    stopped_tokens_new = [raw for raw in stopped_tokens if not raw in remove_words]
    
    # lemmatize tokens
    lemma_tokens = [lemmatizer.lemmatize(tokens) for tokens in stopped_tokens_new]
    
    # remove word containing only single char
    new_lemma_tokens = [raw for raw in lemma_tokens if not len(raw) == 1]
    
    # add tokens to list
    texts.append(new_lemma_tokens)

# sample data
print(texts[0])
#%%
# ## Visualization of top words after cleaning
# Word frequencies visualization in the dataset after cleaning
from collections import Counter
word_list = [item for sublist in texts for item in sublist]
#word_list = texts


counts = dict(Counter(word_list).most_common(50))

labels, values = zip(*counts.items())

# sort your values in descending order
indSort = np.argsort(values)[::-1]

# rearrange your data
labels = np.array(labels)[indSort]
values = np.array(values)[indSort]

indexes = np.arange(len(labels))
#all_words = review_df['CSCF_TEXT'].str.split(expand=True).unstack().value_counts()
data = [go.Bar(
            x = labels,
            y = values,
            marker= dict(colorscale='Jet',
                         color = all_words.values[2:100]
                        ),
            text='Word counts'
    )]

layout = go.Layout(width=1200, height=500,
    title='Top 50  Word frequencies in the dataset (Cleaned)'
)

fig = go.Figure(data=data, layout=layout)

pyo.iplot(fig, filename='basic-bar')
#%%
#For each documents, we should find out how frequently each term occurs, for this we need DTM (document-term matrix)
# # Create the term dictionary
#- assigning a unique integer id to each unique token while also collecting word 
# counts and relevant statistics   
## assign unique id to the unique token
dictionary = corpora.Dictionary(texts)
# Create the Term-Document Matrix
# - convert tokenized documents into TDM    
# convert dictionary into bag of words
corpus = [dictionary.doc2bow(texte) for texte in texts]
#Here corpus, is a list of vectors equal to the number of documents. In each document
#  vector is a series of tuples, where the first element is the term id and second 
# one is term frequency.  
# Let's check how corpus looks like
print(corpus[0])
# let's check the term id for each
print(dictionary.token2id)

#In the above example 'locate' has id 18 and then we have tupple (18,2) that locate 
# is repeated twice in the document.
# Creating the LDA Model
# Here we can use any number of topic you like to create. I just choose for the first 
# iteration to use used pre-determined number of topics to be 6. 
 #However we should calculate perplexity (entangle state) to find the optimum number of topic
 
#%%
 # we can put any number of topic: here I am using only 6 topics, bigger the passes no more accurate model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=6, id2word = dictionary, passes=20);
# we can put any number of topic: here I am using only 6 topics, bigger the passes no more accurate model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=6, id2word = dictionary, passes=20);
#Here we can clearly see that generated each topic is separated by prenthesis and within each topic there
# are three most probable words are given in that topic.
import pprint
pprint.pprint(ldamodel.top_topics(corpus,topn=5));
#%%
import pyLDAvis as ps
from pyLDAvis import enable_notebook, gensim
# Plotting the topic modeling
#pyLDAvis.enable_notebook()
#pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
ps.enable_notebook()
ps.gensim.prepare(ldamodel, corpus, dictionary)
#%%[markdown]
#You may save this model to the disk and re-load pre-trained model for the unseen documents
# For detail: 
https://radimrehurek.com/gensim/models/ldamodel.html