from pythainlp.corpus.common import thai_words
from pythainlp.corpus import thai_stopwords
from pythainlp.util import dict_trie
from pythainlp import word_tokenize
from gensim.models import CoherenceModel
import gensim
import gensim.corpora as corpora
import pyLDAvis.gensim
import pandas as pd
import re



def read_excel(excel_directory, sheet_name):
    df = pd.read_excel(f'.\{excel_directory}', sheet_name=sheet_name)

    return df


def filter(df, column_tofilter, select_value):
    # check is there a column that user want to filter
    if column_tofilter != '':
        df_filter = df[df[column_tofilter] == select_value]
        df_filter = df_filter.reset_index()
    else:
        df_filter = df

    return df_filter


def combine_column(df, column_tocombine):
    # combine sentenct from each column
    df_sentences = pd.DataFrame(columns=['sentences'])
    df_sentences['sentences'] = df[column_tocombine].apply(" ".join, axis=1)

    return df_sentences


def sentence_preprocessing(df):
    # text preprocessing
    data = (df['sentences'].str.lower()).values.tolist()
    data_re = [re.sub('(\n|\s{2})', '', sent) for sent in data]  # remove \n
    data_re = [re.sub('\S*@\S*\s?', '', sent) for sent in data_re]  # remove email
    data_re = [re.sub(r"\d", '', sent) for sent in data_re]  # remove number

    return data_re

def sentence_to_words(sentences, word_config):
    custom_th_words = set(thai_words())
    custom_th_words.update(word_config)
    trie = dict_trie(dict_source=custom_th_words)
    for sentence in sentences:
        yield (word_tokenize(str(sentence), engine='newmm', custom_dict=trie, keep_whitespace=False))


def remove_stopwords(words, add_stopword):
    stopwords = list(thai_stopwords())
    list_wo_stop = []
    for add in add_stopword:
        stopwords.append(add)
    for sentenct in range(0, len(words)):
        without_stop_words = []
        for word in words[sentenct]:
            if word not in stopwords:
                without_stop_words.append(word)
        list_wo_stop.append(without_stop_words)
    return list_wo_stop

def topics_matching(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()
    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return sent_topics_df


def topic_visualize(lda_model, corpus, id2word, Ngroup, visualize_save_name):
    # Topic presentation
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(vis, '{}_{}_topics.html'.format(visualize_save_name, Ngroup))


def topic_tableformat(df_topic, df_input, add_columns):
    # Formating output Excel
    df_dominant_topic = df_topic.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    # Add column from origin Excel to result Excel
    for column in add_columns:
        df_dominant_topic[column] = df_input[column]

    return df_dominant_topic

def topic_modeling(df_input, sentences, words, group_min, group_max, visualize_save_name, add_columns):

    # Topic Modeling
    id2word = corpora.Dictionary(words)
    texts = words
    corpus = [id2word.doc2bow(text) for text in texts]

    # Find optimize number of group
    dfs = {}
    for Ngroup in range(group_min, group_max + 1):
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=Ngroup,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=100,
                                                    passes=10,
                                                    alpha='auto',
                                                    per_word_topics=True)
        # create dataframe for Mapping topic with Text each Ngroup
        sentence_topics_df = topics_matching(ldamodel=lda_model, corpus=corpus, texts=sentences)

        # topic visualization
        topic_visualize(lda_model, corpus, id2word, Ngroup, visualize_save_name)

        # df formatting
        topics_formatted_df = topic_tableformat(sentence_topics_df, df_input, add_columns)

        # collecting dataframe of each group
        dfs[Ngroup] = topics_formatted_df

    return dfs


def topic_2excel(dfs, excel_save_name):
    # write mapping topic to excel
    writer = pd.ExcelWriter('{}.xlsx'.format(excel_save_name))
    for sheet in range(group_min, group_max + 1):
        dfs[sheet].to_excel(writer, sheet_name='{}_topics'.format(sheet))
    writer.save()
    writer.close()


if __name__ == '__main__':

    ############# Config Variable ####################################################

    # input target Excel name here
    excel_directory = 'svd_Q1_Q2.xlsx'
    # input target sheet name here
    sheet_name = 'software_ticket'

    # input column name that want to filter (leave it blank if user don't want to filter)
    column_tofilter = 'Technician'
    # input select value that want to group for analysis
    select_value = 'KHATHATHEP CHANKASAME'

    # input column that want to combine to sentences
    column_tocombine = ['Technician', 'Subject', 'Description']

    # list of word that user want software to that it CAN'T separate to individual word
    word_config = ['ดาต้า', 'ปริ้น', 'ใบขน', 'ปรากฎ', 'ดราฟ', 'มินีแบ', 'microsoft 365', 'technical soft', 'tecnical soft',
               'ซอฟแวร์', 'อินวอ', 'not assigned', 'auto decl', 'ชีท', 'back up', 'ใบหัก', 'log in', 'ใบกำกับ',
               'auto send', 'รีสตาร์ท']

    # input word that user want to cut off analysis
    add_stopwords = ["not assigned", "nan", "ka", "นะคะ", "-", "_", "", " ", "/", "//", "(", ")", ">", "<", ">>", "<<", "'",
               '.', ':', '...', ',', '.,']

    # input range of group that user want software generate topics grouping
    group_min = 2
    group_max = 3

    # input output save name !! without file extension !!
    visualize_save_name = 'vis_vis' # without file extension
    excel_save_name = 'exc_exc' # without file extension

    # add column from origin Excel that user want to add in output Excel
    add_columns_to_finalexcel = ['Group', 'Request ID']

    ##################### Analysis Part ###############################################

    # read Excel
    df = pd.read_excel(excel_directory, sheet_name=sheet_name)

    # filter table
    df_filter = filter(df, column_tofilter, select_value)

    # combine target column to be sentence
    df_sentences = combine_column(df_filter, column_tocombine)

    # text preprocessing
    sentences = sentence_preprocessing(df_sentences)

    # sentences to words
    data_words = list(sentence_to_words(sentences, word_config))

    # remove stop word from data_words
    data_words_wo_stopword = remove_stopwords(data_words, add_stopwords)

    # create Topic Modeling
    df_topics = topic_modeling(df_filter,
                               sentences,
                               data_words_wo_stopword,
                               group_min,
                               group_max,
                               visualize_save_name,
                               add_columns_to_finalexcel)

    # save topic grouping result as Excel
    topic_2excel(df_topics, excel_save_name)

