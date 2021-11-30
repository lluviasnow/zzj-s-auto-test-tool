import os
import numpy as np
import nltk
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from gensim.models.word2vec import Word2Vec
from selenium import webdriver  # 浏览器驱动器
from selenium.webdriver.common.by import By  # 定位器
from selenium.webdriver.common.keys import Keys  # 键盘对象
from selenium.webdriver.support import expected_conditions as EC  # 判断器
from selenium.webdriver.support.wait import WebDriverWait  # 浏览器等待对像
import time
from tqdm import tqdm

#四大功能模块：数据集预处理，word2vec训练，RNN训练和Selenium自动化测试
#通过控制台的输入（上文）和训练好的模型预测下文，然后唤起浏览器在指定网址搜索生成的内容。
# 点击运行，在word2vec训练和RNN训练完成后，输入前馈文本“movie”或者可能存在的标签例如“I want  to look for movie”。
# 然后点击回车，selenium会自动唤起Google Chrom浏览器（运行的电脑需要有这个浏览器且安装了对应版本的Chrome Driver才能达到效果）
# 进入豆瓣电影网页，输入生成的预测值，并点击确认获得期望的搜索结果。程序最后在页面停留一定时间后会自动关闭浏览页面并结束运行。
#把文本读入
raw_text = ''
for file in os.listdir("input"):
	if file.endswith(".txt"):
		raw_text+=open("input/"+file,errors='ignore',encoding='utf-8').read()+'\n\n'   #读入多个文件

raw_text = raw_text.lower()
sentensor = nltk.data.load('tokenizers/punkt/english.pickle')#加载语料库
tokens = sentensor.tokenize(raw_text)  #将他们分解为一个个词法元素
corpus = []
tot=0
for tok in tqdm(tokens):#遍历，加载数据集
	tot=tot+1
	#print(nltk.word_tokenize(sen))
	corpus.append(nltk.word_tokenize(tok))
	if tot>500:#受内存困扰只能选择一些最常见,即前一定数目的加入训练！如果电脑能带动就把这两行注释掉
		break

print(len(corpus))
print(corpus[:3])
#语料库里添加一些元素
corpus.append(nltk.word_tokenize('movie Harry Potter')) #这些是我手动添加的
corpus.append(nltk.word_tokenize('movie Harry Potter movie Harry Potter movie Harry Potter'))
corpus.append(nltk.word_tokenize('movie Harry Potter'))
corpus.append(nltk.word_tokenize('movie Harry Potter'))
corpus.append(nltk.word_tokenize('movie Harry Potter movie Harry Potter movie Harry Potter'))

#正式进入word2vec模型训练的部分

w2v_model = Word2Vec(corpus,vector_size=128, window = 5, min_count = 1, workers = 4)

# 把源数据变成一个长长的x，好让LSTM学会预测下一个单词

raw_input = [item for sublist in corpus for item in sublist]

text_stream = []
vocab=list(w2v_model.wv.index_to_key)
for word in raw_input:
	if word in vocab:
		text_stream.append(word)
#len(text_stream)

#这里我们的文本预测积就是，给出了前面的单次以后，下一个单词是谁？
#举个例子就是hello from the other ,给出side; I want to look for movie, 给出Harry Potter

#构造训练测试集
#x 是前置字母们 y 是后一个字母

seq_length=4
x=[]
y=[]

for i in range(0, len(text_stream)-seq_length):
	given = text_stream[i:i+seq_length]
	predict = text_stream[i+seq_length]
	x.append(np.array([w2v_model.wv[word] for word in given]))
	y.append(w2v_model.wv[predict])


# x=np.array(x)
# y=np.array(y)
x = np.reshape(np.array(x),(-1,seq_length,128))
y = np.reshape(np.array(y),(-1,128))

#现在我们已经有了一个input的数字表达（w2v），我们要把它变成RNN需要的数组格式： [样本数，时间步伐，特征]
#对于output，我们直接用128维的输出
#正式进入RNN模型训练部分
model = Sequential()#首先建立一个modle，然后调整参数，添加恰当格式的数据集
#加入LSTM的模型，LSTM是RNN的一种，由于有现成的接口我们选择他。
model.add(LSTM(256, recurrent_dropout=0.2,input_shape=(seq_length,128)))
model.add(Dropout(0.2))
model.add(Dense(128,activation='sigmoid'))
model.compile(loss='mse',optimizer='adam')
#训练模型
model.fit(x,y,epochs=1000,batch_size=4096)#一些参数，为了快一点看到结果也可以调节。
#模型训练完成，下面正式进入文本生成+selenium预测部分
#测试RNN：
def predict_next(input_array):#预测下一个单词的核心函数，利用训练好的模型和前文进行predict
	input_array=np.array(input_array)
	x = np.reshape(input_array,(-1,seq_length,128))
	y = model.predict(x)
	return y

def string_to_index(raw_input): #在进行predict next之前要先转换数据格式
	raw_input = raw_input.lower()
	input_stream = nltk.word_tokenize(raw_input)
	res = []
	for word in input_stream[(len(input_stream)-seq_length):]:
		res.append(w2v_model.wv[word])
	return res

#利用w2v_model将结果转换为单词
def y_to_word(y):
	word = w2v_model.wv.most_similar(positive = y,topn=1)
	return word

#生成电影名称
def generate_input(init,rounds=2): #将电影名称设为两位，因为这是最常见的
	in_string = init.lower() #统一转为小写，大小写不同本质是一个词，不应该让他影响我们的结果
	for i in range(rounds):
		n=y_to_word(predict_next(string_to_index(in_string)))
		in_string +=' '+n[0][0]
	return in_string

ini=input() #从控制台得到“上文”以预测下文
my_input = generate_input(ini)#通过输入得到生成的句子
print(my_input)
browser = webdriver.Chrome() #准备好要唤起的浏览器
splitIn=my_input.split()
len=len(splitIn)
splitIn=splitIn[len-2]+" "+splitIn[len-1]#设定电影名称长度为两位
try:
    # 浏览器对象豆瓣电影
    browser.get("https://movie.douban.com/")
    # 查找id为 'kw'的标签，即输入框
    #inputs = browser.find_element_by_id("kw")
    inputs=browser.find_element_by_tag_name("input")
    print(splitIn)
    #inputs.send_keys("Harry Potter")
    inputs.send_keys(splitIn)
    # 在输入框中填入'Python'
    #inputs.send_keys("Python")
    # '按下'回车键,第一种`	`	1
    inputs.send_keys(Keys.ENTER)
    # 点击,第二种
    browser.find_element_by_id("su").click()
    # 创建一个等待对像，超时时间为10秒，调用的时间间隔为0.5
    wait = WebDriverWait(browser, 10, 0.5)
    # 每隔0.5秒检查一次，直到页面元素出现id为'content_left'的标签
    wait.until(EC.presence_of_all_elements_located((By.ID, "content_left")))
except Exception as e:
    print(e)
else:
    # 打印请求的url
    print(browser.current_url)
    # 打印所有cookies
    print(browser.get_cookies())
finally:
    # 等待10秒
    time.sleep(10)
    # 关闭浏览器对象
    browser.close()