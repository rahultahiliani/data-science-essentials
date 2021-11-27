
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
df.sample()

sr = pd.crosstab(df['Sex'],df['Survived'])



ax = sb.countplot(x ='Sex',hue='Survived', palette='Set1',data=df)
ax.set(title='Survivors',xlabel='Sex',ylabel='Numbers')
plt.show()




fp = sb.factorplot(x='Pclass',y='Survived',hue='Sex',data=df,aspect=0.9,size=3)

fp_Ebbark = sb.factorplot(x='Embarked',y='Survived',hue='Sex',data=df,aspect=0.9,size=3)


















