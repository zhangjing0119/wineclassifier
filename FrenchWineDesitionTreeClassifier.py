#coding:gbk
"""
���þ������㷨���з���
���ߣ�ֲ�������ž�
���ڣ�2020/05/13
"""
import pandas as pd           # ������Ҫ�õĿ�
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sb
#%matplotlib inline
# ��������
df = pd.read_csv('frenchwine.csv')
df.columns = ['     species', 'alcohol', 'malic_acid', 'ash', 'alcalinity ash','magnesium']
# �鿴ǰ5������

# �鿴����������ͳ����Ϣ
df.describe()
print(df.describe())



plt.figure(figsize=(20, 10)) #����seaborn�������������Ʒ�ֲ�ͬ����ͼ
for column_index, column in enumerate(df.columns):
    if column == '     species':
        continue
    plt.subplot(3, 2, column_index)
    sb.violinplot(x='     species', y=column, data=df)
plt.show()

# ���ȶ����ݽ����з֣������ֳ�ѵ�����Ͳ��Լ�
from sklearn.model_selection import train_test_split #����sklearn���н�����飬����ѵ�����Ͳ��Լ�
all_inputs = df[['alcohol', 'malic_acid',
                             'ash', 'alcalinity ash','magnesium']].values
all_species = df['     species'].values

(X_train,
 X_test,
 Y_train,
 Y_test) = train_test_split(all_inputs, all_species, train_size=0.85, random_state=1)#85%������ѡΪѵ����

# ʹ�þ������㷨����ѵ��
from sklearn.tree import DecisionTreeClassifier #����sklearn���е�DecisionTreeClassifier������������
# ����һ������������
decision_tree_classifier = DecisionTreeClassifier()
# ѵ��ģ��
model = decision_tree_classifier.fit(X_train, Y_train)
# ���ģ�͵�׼ȷ��
print(decision_tree_classifier.score(X_test, Y_test)) #st��ʵ������
test=[[13.52,3.17,2.72,23.5,97],[12.42,2.55,2.27,22,90],[13.76,1.53,2.7,19.5,132]]  #��������test��ֱ����������
list=model.predict(test)  #���ԣ�������������3�����ݽ��в��ԣ���ȡ3��������Ϊģ�͵�����
for x in range(0,len(list)):
	if list[x]=="Zinfandel":
	     list[x]="�ɷ���"
	elif list[x]=="Syrah":
		 list[x]="����"
	else:
		 list[x]="��ϼ��"   #��Ӣ�ķ��������
print(list)#������ԵĽ���������ģ��Ԥ��Ľ��
 
