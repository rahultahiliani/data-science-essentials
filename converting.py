import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df_train = pd.read_csv('train.csv')


def title(name):
	if '.' in name:
		return name.split(',')[1].split('.')[0].strip()
	else:
		return 'No title Exists'


unique_titles =sorted(set([x for x in df_train['Name'].map(lambda x: title(x))]))

print(len(unique_titles) ,':' ,unique_titles)


def compact_titles(x):
	title_new = x['Title']
	if title_new in ['Capt', 'Col', 'Major']:
		return 'Officer'
	elif title_new in ['Mr','Master','Don','the Countess','Sir', 'Ms','Jonkheer', 'Dona', 'Lady']:
		return 'Royale People'
	elif title_new in ['Mrs','Master','Miss''the Countess','Mme','Lady']:
		return 'Mrs'
	elif title_new in ['Miss','Mrs','Mlle', 'Ms']:
		return 'Miss'
	else:
		return title_new


df_train['Title'] = df_train['Name'].map(lambda x: title(x))

df_train['Title'] = df_train.apply(compact_titles, axis=1)

print(df_train.Title.value_counts())
df_train['Age'].fillna(df_train['Age'].median(), inplace =True)
df_train['Fare'].fillna(df_train['Fare'].median(), inplace =True)
df_train['Embarked'].fillna("S" , inplace =True)
df_train.drop('Name', axis=1, inplace=True)
df_train.drop('Ticket', axis=1, inplace=True)
df_train.drop('Cabin', axis=1, inplace=True)
df_train.Sex.replace(('male', 'female'), (0, 1), inplace=True)
df_train.Embarked.replace(('S', 'Q', 'C'), (0, 1, 2), inplace=True)
df_train.Title.replace(('Mrs', 'Miss', 'Royale People', 'Dr', 'Rev', 'Officer','Royale People'), (0, 1, 2, 3, 4, 5, 6), inplace=True)



print(df_train)

import pandas as pd
from sklearn.model_selection import train_test_split

# pd.set_option('display.max_columns', None)


df_train = pd.read_csv('train.csv')

df_train.sample(5)


y = df_train['Survived']
x = df_train.drop(['Survived', 'PassengerId'], axis=1)





x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier()
fit = rf.fit(x_train,y_train)
y_predicted = rf.predict(x_test)
#
acc_rf = round(accuracy_score(y_predicted,y_test)*100,2)
print(f'Accuracy of Model is :{acc_rf}')
pickle.dump(rf,open('my_titanic_model.sav','wb'))





