import pythainlp
from pythainlp import word_tokenize
from pythainlp.corpus import get_corpus # for getting stopwords
import pandas as pd
import wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import re

df = pd.read_excel('D:\Luk_works\python_projects\SVD_project\svd_sample1_prepare.xlsx', sheet_name='trainning_data')
df_Sysfreight = df[df['Calss'] == 'Other Software']
df_Sysfreight['word'] = df_Sysfreight['header'].astype(str) + ' ' + df_Sysfreight['description'].astype(str)
all_words_Sysfreight = ' '.join(df_Sysfreight['word']).lower()
words = word_tokenize(all_words_Sysfreight)
words_str_join = ' '.join(words).lower().strip()
words_str_join = re.sub('(\n|\s{2})', '', words_str_join)

stopwords = pythainlp.corpus.thai_stopwords()
stopwords = set(list(stopwords)).union({'นี้', 'อัน', 'แต่', 'ไม่'})

wordcloud = WordCloud(
font_path='c:/windows/fonts/cordia.ttc',
    regexp=r"[\u0E00-\u0E7Fa-zA-Z']+",
    stopwords=stopwords,
    width=2000, height=1000,
    prefer_horizontal=1,
    max_words=30,
    colormap='tab20c',
    background_color = 'white').generate(words_str_join)
plt.figure(figsize = (10, 9))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()