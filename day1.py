"""code from day2 of ASP 2020 course"""
# Muss das in die Python Console kopieren und da executen
import pandas as pd  # kürze Pandas als pd ab

FNAME = "http://www.stat.ucla.edu/~rgould/datasets/twins.dat"  # schreibe GROß um zu zeigen, dass es ein Konstant ist

df = pd.read_csv(FNAME, sep='\t')
# define the dataframe, which is in csv, columns separated by a tab, study is on the effect of education on income in twins


df.shape #Dimensions
df.head() #first 5 lines by default, otherwise put in another number
df.tail(7) #Last 7 lines
df.columns #List of variables
df.describe()#Summary statistics

#Selecting columns
df["DEDUC1"]#Column by column name
df[["AGE","LHRWAGEH"]]
df.iloc[:,5:7]#i is an index, use the number of columns, remeber it starts with 0, want all rows, so put only the ':'

#Selecting rows
df.loc[0]#Row by index name (also accepts lists)
df.iloc[0] #Row by row nuber (also accepts lists)

#Selecting values
df.loc[18,"AGE"] #Number of row and column
df.iloc[18,2] #Index of row and column, can do here -1 for the last observation, -2 for the second last etc.

# Get info
df.info() #objects (strings), integers(number, 1, 2, 3), float (number with Nachkommastelle)

#Changing dtypes
df["DMARRIED"]=df["DMARRIED"].astype(bool)
df["WHITEH"]=df["WHITEH"].astype("category")
df["LHRWAGEH"]=pd.to_numeric(df ["LHRWAGEH"], errors="coerce") #if no value provided turn into NAN

df.info(memory_usage=True) #very detailed way of showing df

#saving memory by optimizing dtypes
bools = ['WHITEH', 'MALEH', 'WHITEL', 'MALEL']
df[bools] = df[bools].astype(bool)
df['DMARRIED'] = df['DMARRIED'].astype('int8')
df.info(memory_usage=True)

# Boolean indexing where you select rows ad columns based on whether a condition is true or false
df[df['AGE'] > 20]
df[(df['AGE'] > 20) & (df['WHITEL'] == 1)]
df[~(df['AGE'] > 20)]
values = (20,21,22,23)
df[df['AGE'].isin(value)]
## not equal to is ~ before paranthesis or !=
#
df[df['WHITEL'] == 0]
df[df['WHITEL'] == 0].shape
sum(df['WHITEL'] == 0)
#11
df[~(df['DEDUC1'] == 0) & (df['WHITEH'] == 1)]
# 89
df[~(df['WHITEL'] == df['WHITEH'])]
df[(df['WHITEL'] != df['WHITEH'])]
#3

#mean age of twins whose L sibling is a non-white male who is married
mask= ((df['WHITEL'] != 1) & (df['MALEL'] != 1) & (df['DMARRIED'] == 1))
df[mask]['AGE'].mean()

df.groupby(['MALEL']).mean()

df.groupby(['MALEL']).agg(['mean', 'min'])
df.groupby(['MALEL'])[['AGE' , 'DEDUC1']].agg(['mean', 'min'])

# Plotting with matplotlib and pandas and with seaborn
# Seaborn is specifically made for statistical plotting
# long versus wide data , be careful about it

import matplotlib.pyplot as plt

# visualization with pandas
df.plot.scatter(x='AGE', y='DEDUC1')

df.plot.line(x='AGE', y='AGESQ', marker='x', color='green')
import matplotlib.pyplot as plt

fig, ax = plt.subplots() # blank canvas
df.plot.scatter(x='AGE', y='AGESQ', marker='x', color='green', ax=ax)
#ax.set(ylabel='Age squared', title='Age vs. Age squared')
ax.set(xlabel='Age', ylabel='Age squared', title='Age vs. Age squared')

fig.savefig('fig1.png')

# creating multiple plots in one figure
fig, axes = plt.subplots(1,2)  #  rows and columns
# axes is now a list so we need to specify the elements
df.plot.scatter(x='AGE', y='AGESQ', ax=axes[0])
df.plot.scatter(x='AGE', y='AGESQ', ax=axes[1])


fig, axes = plt.subplots(1,2, sharey=True)  #  rows and columns
# axes is now a list so we need to specify the elements
df.plot.scatter(x='AGE', y='AGESQ', ax=axes[0])
df.plot.scatter(x='AGE', y='AGESQ', ax=axes[1], color='green')

fig, axes = plt.subplots(2,2, sharey=True)  #  rows and columns
# axes is now a list so we need to specify the elements
df.plot.scatter(x='AGE', y='AGESQ', ax=axes[0,0])
df.plot.scatter(x='AGE', y='AGESQ', ax=axes[1,0], color='green')
df.plot.scatter(x='AGE', y='AGESQ', ax=axes[0,1])
df.plot.scatter(x='AGE', y='AGESQ', ax=axes[1,1], color='green')


# SEABORN
# Has three general plotting families #  relplot() for scatter and line
                                      #  catplot() for categories
                                      #  regplot() linear relationships
#panda has the dataframe format built in but seaborn does not so you need to specify the data source
import seaborn as sns
sns.set(style='white', palette='Set1')

fmri = sns.load_dataset('fmri')
fmri.head()
fmri['event'].unique()  # to see the different values of a variable
sns.relplot(x='timepoint', y='signal', data=fmri, kind='scatter')

sns.relplot(x='timepoint', y='signal', hue='event', data=fmri, kind='scatter', style='event')

sns.relplot(x='timepoint', y='signal', hue='event', data=fmri, kind='line', style='event')

sns.relplot(x='timepoint', y='signal', hue='event', data=fmri, kind='line', style='event', ci='sd', markers=True, col='region')
# col is for multiple columns , could also write row for multiple rows
sns.relplot(x='timepoint', y='signal', hue='event', data=fmri, kind='line', style='event', ci='sd', markers=True, row='region')

sns.catplot(x='region', y='signal', data=fmri)

sns.catplot(x='region', y='signal', data=fmri, kind='violin')
sns.catplot(x='region', y='signal', data=fmri, kind='box')

sns.regplot(x='timepoint', y='signal', data=fmri)

# Heatmaps
titanic = sns.load_dataset('titanic')
titanic
# plot gender ratio by class
# for that I do a cross tabulation

cross = pd.crosstab(titanic['class'], titanic['sex'])
cross
sns.heatmap(cross, annot=True)

sns.heatmap(cross, annot=True, cmap='viridis')

# Debugging
x = '90'
y = 100
z=x+y
z
# Typeerror

currencies = ['dollar', 'euro']
print(currency)
# Nameerror

#value error
int('9.0')

marks = [1,1,4,3,6]
print(marks[5])
#index error

marks = ['Germany','Japan','UK']
print(marks['France'])
#Keyerror

my_list = 'fcbead'
my_list.sort()
#Attributeerror

# try and except block can be use
try:
    #do something
except Error
    # do something
