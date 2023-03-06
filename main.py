# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np

import pandas as pd


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def test1():
    a = [3, 2, 6, 1, 9, 0]
    a.sort()
    print(a)
    data = {i: np.random.randn() for i in range(7)}
    print(data)
    import bisect
    print(bisect.bisect(a, 6))
    for i, val in enumerate(a):
        print(i, val)
    print(sorted("afafsgsg"))
    seq = ["1", "2", "3"]
    seq2 = ["one", "two", "three"]
    zipped = zip(seq, seq2)
    print(list(zipped))
    print(list(reversed(range(10))))
    empty_dict = {}
    d1 = {'a': "somr", 'b': [1, 23, 4]}
    print(d1)
    print(d1['b'])
    print('b' in d1)
    print(hash((2, 3)))
    ss = {2, 2, 2, 3, 3}
    print(ss)
    strs = ['a', 'as', 'bas', 'cat', 'dobve']
    sss = [x.upper() for x in strs if len(x) > 2]
    print(sss)
    lst = {1, 2, "DDD"}
    for i in iter(lst):  # iter方法，创建迭代器的指针,指针指向一个list
        print(i)
    ss = []
    for i in lst:
        ss.append(i)
    print("ss", ss)
    print(list(iter(lst)))


equiv_anon = lambda x: x * 2


def readfile():
    filepath = "1.txt"
    with open(filepath, "r") as f:
        lines = [x.rstrip() for x in f]  # rstrip()方法，去除尾随的换行和空格
        print(lines)


def squares(n=10):
    print('Generating squares from 1 to {0}'.format(n ** 2))
    for i in range(1, n + 1):
        yield i ** 2  # yield 返回一个生成器，替代return，和iter方法生成的效果相同
        # 返回 [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]


def test2():
    arr = np.empty((8, 4))
    for i in range(8):
        arr[i] = i

    print(arr)
    print(arr[[4, 3, 0, 6]])
    arr = np.arange(32).reshape((8, 4))
    print(arr)
    print(arr[[1, 5, 7, 2], [0, 3, 1, 2]])
    arr = np.random.randn(6, 3)
    print(arr)
    print(arr.T)
    arr = np.arange(10)
    print(np.sqrt(arr))
    points = np.arange(-5, 5, 0.01)
    print(points)
    xs, ys = np.meshgrid(points, points)
    print(xs)
    print(ys)
    z = np.sqrt(xs ** 2 + ys ** 2)
    names = np.array(['bob', 'joe', 'will', 'bob', 'will'])
    print(np.unique(names))
    obj = pd.Series([4, 7, -5, 4])
    print(obj)
    sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
    obj3 = pd.Series(sdata)
    print(obj3)
    data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
            'year': [2000, 2001, 2002, 2001, 2002, 2003],
            'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
    frame = pd.DataFrame(data)
    print(frame)
    print(frame['state'])
    frame['es'] = frame['year'] == 2000
    print(frame)
    print()
    obj = pd.Series(range(3), index=['a', 'b', 'c'])
    index = obj.index
    print(obj)

    frame = pd.DataFrame(np.arange(9).reshape((3, 3)),
                         index=['a', 'c', 'd'],
                         columns=['Ohio', 'Texas', 'California'])
    frame2 = frame.reindex(['a', 'b', 'c', 'd'])
    print(frame2)
    # frame2.drop(['a', 'b'], inplace=True)  # inplace 选项，直接修改所在frame，不产生新的frame
    print(frame2)
    frame2['a':'d'] = 6
    print(frame2)
    ser2 = pd.Series(np.arange(3.), index=['a', 'b', 'c'])
    print()
    print(ser2)
    print("-1\n", ser2[-1])
    print("iloc\n", ser2.iloc[:1])


def test3():
    frame = pd.DataFrame(np.arange(12.).reshape((4, 3)),
                         columns=list('bde'),
                         index=['Utah', 'Ohio', 'Texas', 'Oregon'])
    series = frame.iloc[0]  # 取一行
    print("frame:\n", frame)
    print("series:\n", series)
    print(frame - series)  # DataFrame和np.series相减，和np的广播机制类似
    frame = pd.DataFrame(np.random.randn(4, 3), columns=list('bde'),
                         index=['Utah', 'Ohio', 'Texas', 'Oregon'])
    print(frame)
    f = lambda x: x.max() - x.min()
    f2 = lambda x: np.cumsum(x)  # 累加方法，等于前面的数据的累加
    fff = frame.apply(f2, axis='columns')  # DataFrame apply方法，应用lambda表达式
    print(fff)
    format = lambda x: '%.2f' % x
    frame = frame.applymap(format)  # applymap格式化字符串，不得使用apply方法，会报错
    print(frame)
    print()
    obj = pd.Series([7, -5, 7, 4, 2, 0, 4])
    print(obj.rank())  # 给出各组平均排名，得出的是排名，会出现.5的情况，奇奇怪怪的，说明有重复数据
    print("\n")
    df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5], [np.nan, np.nan], [0.75, -1.3]],
                      index=['a', 'b', 'c', 'd'], columns=['one', 'two'])
    print(df)
    print(df.describe())


def bit():
    # HDF5 压缩格式，算是二进制  使用PyTables或h5py,pandas访问
    frame = pd.DataFrame({'a': np.random.randn(100)})
    # print(frame)
    # store = pd.HDFStore('mydata.h5')  # store 返回指针
    # store['obj1'] = frame  # 存放全部
    # store['obj1_col'] = frame['a']  # 存放a列
    # print(store)
    # print(store['obj1'])
    # print(store['obj1_col'])
    # store.put('obj2', frame, format='table')
    # print(store.select('obj2', where=['index >= 10 and index <= 15']))
    frame.to_hdf('mydata.h5', 'obj3', format='table')  # 使用'table'模式转换DataFrame为hdf5
    print(pd.read_hdf('mydata.h5', 'obj3', where=['index <5']))


def web_request():
    import requests  # 可以使用request使用web中的API接口
    url = 'https://api.github.com/repos/pandas-dev/pandas/issues'
    resp = requests.get(url)
    print(resp)
    data = resp.json()
    print(data)
    issues = pd.DataFrame(data, columns=['number', 'title', 'labels', 'state'])
    print(issues)


def sql():
    import sqlite3
    # 创建数据库
    # query = """CREATE TABLE test
    #             (a VARCHAR(20),b VARCHAR(20),
    #             c REAL, d INTEGER
    #             );
    #         """
    # con = sqlite3.connect("mydata.sqlite")
    # con.execute(query)
    # con.commit()

    # 插入数据
    # data = [('Atlanta', 'Georgia', 1.25, 6),
    #         ('Tallahassee', 'Florida', 2.6, 3),
    #         ('Sacramento', 'California', 1.7, 5)]
    # con = sqlite3.connect("mydata.sqlite")
    # stmt = "INSERT INTO test VALUES(?,?,?,?)"
    # con.executemany(stmt,data)

    # 基本查询
    # con = sqlite3.connect("mydata.sqlite")
    # cursor = con.execute('select * from test')
    # rows =cursor.fetchall()
    # print(rows)

    # 查询
    con = sqlite3.connect("mydata.sqlite")
    cursor = con.execute('select * from test')
    rows = cursor.fetchall()
    print(rows)
    print(cursor.description)
    frame = pd.DataFrame(rows,columns=[x[0] for x in cursor.description])
    print(frame)

def nan_fliter():
    from numpy import  nan as NA
    data = pd.Series([1,NA,3,5,NA,7])
    # data = data.dropna() # dropna 新建一个新的pd,不覆盖。去除nan数据
    # print(data)
    data = data[data.notnull()]
    # print(data)
    data = pd.DataFrame([[1., 6.5, 3.], [1., NA, NA],
                         [NA, NA, NA], [NA, 6.5, 3.]])
    cleaned =data.dropna()  # 丢失所有包含NAN的行
    print(cleaned)


def delete():
    data = pd.DataFrame({'k1':['one','two']*3 + ['two'],'k2':[1,1,2,3,3,4,4]})
    # print(data)
    # print(data.duplicated())
    data=data.drop_duplicates()
    # print(data)
    # data = data.drop_duplicates('k1')
    # print(data)
    data = data.replace(1,111)
    # print(data)
    # np.sign 表示元素的符号，1：，-1： -，0：0
    df = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
        'data1': range(6)})
    print(df)
    a = pd.get_dummies(df['key'])
    print(a)

def str_model():
    val = 'a,b, guido'
    print(val.split(','))
    pieces = [x.rstrip() for x in val.split()]
    print(pieces)
    import re
    text = "foo   bar\t bas \t quad"
    print(re.split('\s+',text)) # \s+  表示一个到多个的空白符(制表\t,空格，换行)

def re_product():
    import re
    text = 'foo   bar \t baz  \tqux'
    ss = re.split('\s+',text)
    print(ss)
    regex = re.compile('\s+')
    ss = regex.split(text)
    print(ss)
    print(regex.findall(text)) # regex 的所有模式

    text = """Dave dave@google.com
    Steve steve@gmail.com
    Rob rob@gmail.com
    Ryan ryan@yahoo.com
    """
    pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'
    #             字母，数字    @   字母数字      2到4个字母
    regex = re.compile(pattern,flags=re.IGNORECASE)
    print(regex.findall(text))

def cch():
    data = pd.Series(np.random.randn(9),index= [
        ['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],
        [1, 2, 3, 1, 3, 1, 2, 2, 3]  # multi Index
                                                ])
    print(data)
    print(data.index)
    print(data['b'])
    print(data[:,2])
    print(data.unstack().stack())  # stack()，unstack()为逆运算
    frame = pd.DataFrame(np.arange(12).reshape((4,3)),
                         index = [['a','a','b','b'],[1,1,2,2]],
                         columns=[['Ohio','Ohio','Colorado'],
                                  ['Green','Red','Green']])
    frame.index.names = ['k1','k2']
    frame.columns.names = ['st','color']
    print(frame)
    print(frame['Ohio'])
    print(frame.swaplevel(i='k1',j='k2'))
    print(frame.sort_index(level=1)) # sort_index(level= )  index列排序，根据index
    print(frame.sum(level ='k1')) # 列级别求和，仍旧有其他的列分类


def hebing():
    df1 = pd.DataFrame({'key':['b','b','a','c','a','a','b'],
                        'data1':range(7)})
    df2 = pd.DataFrame({'key':['a','b','d']
                        ,'data2':range(3)})

    print(df1)
    print("\n")
    print(df2)
    print(pd.merge(df1,df2,on= 'key'))
    # merge数据库合并，on字段表示合并的键名
    df3 = pd.DataFrame({
        'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
        'data1':range(7)})
    df4 =pd.DataFrame({'rkey':['a','b','d'],
                       'data2':range(3)})
    print(pd.merge(df3,df4,left_on = 'lkey',right_on='rkey'))





def plttt():
    import matplotlib.pyplot  as plt
    # data = np.arange(10)
    # print(data)
    # plt.plot(data)
    # plt.show()
    fig = plt.figure() # figure 对象，表示画布对象
    ax1 = fig.add_subplot(2,2,1)
    ax2= fig.add_subplot(2,2,2)
    ax3 =fig.add_subplot(2,2,3)
    plt.show()





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    plttt()
    # readfile()
    # print(list(range(11)))
    # print(list(squares()))
    # gen = squares()
    # for x in gen:
    #     print(x)
    # a= [1,2,2,2]
    # for i in a:
    #     print(i)
    #
    # test1()
    # data = np.random.randn(2,3)
    # print(data)
    # print(data.dtype)
    # print(data.astype(np.int32))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
