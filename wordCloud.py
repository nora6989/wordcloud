from konlpy.tag import Okt
okt = Okt()
okt.pos("삼성전자 글로벌센터 전자사업부",stem=True)

filename = 'C:/Users/ezen/WEEKEND_TENSORFLOW/wordCloud/data/kr-Report_2018.txt'
with open(filename,'r',encoding='UTF-8') as f: # r 은 read파일
    texts = f.read()
texts[:300]

import re
texts = texts.replace('\n','') # 줄바꿈 제거
tokenizer = re.compile('[^ ㄱ-힣]+') #한글과 띄어쓰기를 제외한 모든글자
texts = tokenizer.sub('', texts) #토큰을 제외한 모든 부분제거
texts[:300]


from nltk.tokenize import word_tokenize
nltk.download() 
tokens = word_tokenize(texts)
tokens[:7]

noun_token = []
for token in tokens:
    token_pos = okt.pos(token)
    temp = [txt_tag[0] for txt_tag in token_pos if txt_tag[1] == 'Noun']
    if len(''.join(temp)) >1:
        noun_token.append("".join(temp))
texts = " ".join(noun_token)
texts[:300]

# 불용어 제거
 
with open('./stopwords.txt','r',encoding='UTF-8') as f: # r 은 read파일
    stopwords = f.read()

stopwords = stopwords.split(' ')
stopwords[:10]

texts = [text for text in tokens if text not in stopwords]
# 원본에서 불용어 파일에 존재하지 않는 단어들만 추출하라

import pandas as pd
from nltk import FreqDist
freqtxt = pd.Series(dict(FreqDist(texts))).sort_values(ascending=False)
freqtxt[:25] 

# 판다스를 활용하여 상위 빈도 단어를 추출한다

from konlpy.tag import Okt 
# stem 어간.. 의미를 가지는 단어
# tag 문법.. 명사, 동사,...
okt = Okt()
okt.pos('가치창출')
okt.pos('갤럭시')

#워드 클라우드 출력 

from wordcloud import WordCloud 
wcloud = WordCloud("C:/Users/ezen/WEEKEND_TENSORFLOW/wordCloud/data/D2Coding.ttf",
                   relative_scaling =0.2, 
                   background_color ='white').generate(" ".join(texts))
wcloud

import matplotlib.pyplot as plt 
plt.figure(figsize =(12, 12))
plt.imshow(wcloud, interpolation='bilinear')
plt.axis("off")
plt.show
  
