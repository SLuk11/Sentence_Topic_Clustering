from pythainlp import word_tokenize
from pythainlp.corpus import get_corpus
from wordcloud import WordCloud
import re
import matplotlib.pyplot as plt
import pandas as pd
import pythainlp



def read_excel(excel_directory, sheet_name):
    df = pd.read_excel(f'.\{excel_directory}', sheet_name=sheet_name)

    return df


def filter(df, column_tofilter, select_value):
    # check is there a column that user want to filter
    if column_tofilter != '':
        df_filter = df[df[column_tofilter] == select_value]
    else:
        df_filter = df

    return df_filter


def sentence2words(df, column_tocombine):
    # combine sentenct from each column
    df_sentences = pd.DataFrame(columns=['sentences'])
    df_sentences['sentences'] = df[column_tocombine].apply(" ".join, axis=1)

    # turn sentence into list of words
    all_words = ' '.join(df_sentences['sentences']).lower()
    words = word_tokenize(all_words)
    words_str_join = ' '.join(words).lower().strip()
    words_str_join = re.sub('(\n|\s{2})', '', words_str_join)

    return words_str_join


def generate_stopwords(add_stopwords):
    # get list of word that user want to remove
    stopwords = pythainlp.corpus.thai_stopwords()
    stopwords_added = set(list(stopwords)).union(set(add_stopwords))

    return stopwords_added


def create_wordcloud(words, stopwords):
    # Create Word Cloud image
    wordcloud = WordCloud(
        font_path='c:/windows/fonts/cordia.ttc',
        regexp=r"[\u0E00-\u0E7Fa-zA-Z']+",
        stopwords=stopwords,
        width=2000, height=1000,
        prefer_horizontal=1,
        max_words=30,
        colormap='tab20c',
        background_color='white').generate(words)
    plt.figure(figsize=(10, 9))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()


if __name__ == '__main__':

    ############# Config Variable ####################################################

    # input traget Excel name here
    excel_directory = 'svd_Q1_Q2.xlsx'
    # input traget sheet name here
    sheet_name = 'software_ticket'

    # input column name that want to filter (leave it blank if user don't want to filter)
    column_tofilter = 'Technician'
    # input select value that want to group for analysis
    select_value = 'KHATHATHEP CHANKASAME'

    # input column that want to combine sentences
    column_tocombine = ['Technician', 'Subject', 'Description']

    # input word that user want to cut off analysis
    add_stopwords = ['นี้', 'อัน', 'แต่', 'ไม่']


    ##################### Analysis Part ###############################################

    # read Excel
    df = pd.read_excel(excel_directory, sheet_name=sheet_name)

    # filter table
    df_filter = filter(df, column_tofilter, select_value)

    # generate list of words
    words = sentence2words(df_filter, column_tocombine)

    # generate list of remove words
    stopwords = generate_stopwords(add_stopwords)

    # create word cloud
    create_wordcloud(words, stopwords)

