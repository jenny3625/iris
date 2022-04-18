# iris
以Iris dataset為例，鳶尾花資料集是非常著名的生物資訊資料集之一，取自美國加州大學歐文分校的機器學習資料庫。http://archive.ics.uci.edu/ml/datasets/Iris

# 載入鳶尾花資料集
from sklearn.datasets import load_iris

# 獲取鳶尾花資料集_小資料集獲取 load_*
iris = load_iris()
#資料集屬性描述
print("鳶尾花資料集的返回值：\n", iris)
# 返回值是一個繼承自字典的Bench
print("鳶尾花的特徵值:\n", iris["data"])
print("鳶尾花的目標值：\n", iris.target)
print("鳶尾花特徵的名字：\n", iris.feature_names)
print("鳶尾花目標值的名字：\n", iris.target_names)
print("鳶尾花的描述：\n", iris.DESCR)
#還有對應的feature_names

# 載入鳶尾花資料集_大資料集獲取 fetch_*
from sklearn.datasets import fetch_20newsgroups
news=fetch_20newsgroups()
print(news["data"][0:5])

# 檢視資料分佈 seaborn畫圖的
#用的版本老可能畫圖顯示不出來，所以要加上%matplotlib inline
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
#資料用dataframe儲存,根據特徵值的名字命名
iris_data=pd.DataFrame(data=iris.data, columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
print(iris_data)

#新增目標值(自變數)
iris_data["target"]=iris.target
iris_data

#定義畫圖函式
def iris_plot(data,col1,col2):
    sns.lmplot(x=col1,y=col2,data=data)
    plt.show()
    
iris_plot(iris_data,"Sepal_Length","Petal_Length")

# 傳入目標的類別 target ，和 fit_reg=False 表示不要線
#定義畫圖函式
def iris_plot(data,col1,col2):
    sns.lmplot(x=col1,y=col2,data=data,hue="target",fit_reg=False)
    plt.title("鳶尾花資料展示")
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.show()
iris_plot(iris_data,"Sepal_Length","Petal_Length")

# 資料集的劃分
from sklearn.model_selection import train_test_split
#會返回四個值iris.data拆分成特徵值，iris.target拆分成目標值
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target)
print('訓練集的特徵值{}\n測試集的特徵值{}\n訓練集的目標值{}\n測試集的目標值{}'
      .format(x_train.shape,x_test.shape,y_train.shape,y_test.shape))
      
#隨機數種子，我們這裡傳入2
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.2,random_state=2)
print('測試集的目標值\n',y_test)
#隨機數種子，我們這裡傳入22
x_train1,x_test1,y_train1,y_test1=train_test_split(iris.data,iris.target,test_size=0.2,random_state=22)
print('測試集的目標值\n',y_test1)
x_train2,x_test2,y_train2,y_test2=train_test_split(iris.data,iris.target,test_size=0.2,random_state=22)
print('測試集的目標值\n',y_test2)
      
