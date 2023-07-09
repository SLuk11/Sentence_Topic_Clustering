from pythainlp.corpus.common import thai_words
from pythainlp.util import dict_trie
from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords
import pandas as pd
import re
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
# Visualization tools
import pyLDAvis.gensim

def sent_to_words(sentences):
    custom_th_words = set(thai_words())
    addlist = ['ดาต้า', 'ปริ้น', 'ใบขน', 'ปรากฎ', 'ดราฟ', 'มินีแบ', 'microsoft 365', 'technical soft', 'tecnical soft', 'ซอฟแวร์', 'อินวอ'
        , 'not assigned', 'auto decl', 'ชีท', 'back up', 'ใบหัก', 'log in', 'ใบกำกับ', 'auto send', 'รีสตาร์ท']
    custom_th_words.update(addlist)
    trie = dict_trie(dict_source=custom_th_words)
    for sentence in sentences:
        yield (word_tokenize(str(sentence), engine='newmm', custom_dict=trie, keep_whitespace=False))


def remove_stopwords(lists):
    stopwords = list(thai_stopwords())
    add_stop = ["not assigned", "nan", "ka", "นะคะ", "-", "_", "", " ", "/", "//", "(", ")", ">", "<", ">>", "<<", "'", '.', ':', '...', ',', '.,']
    list_wo_stop = []
    for add in add_stop:
        stopwords.append(add)
    for sentenct in range(0, len(lists)):
        without_stop_words = []
        for word in lists[sentenct]:
            if word not in stopwords:
                without_stop_words.append(word)
        list_wo_stop.append(without_stop_words)
    return list_wo_stop

def format_topics_sentences(ldamodel, corpus, texts):
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


if __name__ == '__main__':
    # get input
    excel_name = input("Xlsx file name: ")
    sheet_name = input("Sheet name: ")
    category = str(input("Service Category: "))
    group_min = int(input("Minimum group of topic: "))
    group_max = int(input("Maximum group of topic: "))
    save_as = input("Save as name: ")

    # read data
    df = pd.read_excel('.\{}'.format(excel_name), sheet_name='{}'.format(sheet_name))
    df_SWincident = df[df['Service Category'] == category]
    df_SWincident = df_SWincident.reset_index()
    df_SWincident['word'] = df_SWincident['Subject'].astype(str) + ' ' + df_SWincident['Description'].astype(str)

    # text preprocessing
    data = (df_SWincident['word'].str.lower()).values.tolist()
    data_re = [re.sub('(\n|\s{2})', '', sent) for sent in data] #remove \n
    data_re = [re.sub('\S*@\S*\s?', '', sent) for sent in data_re] # remove email
    data_re = [re.sub(r"\d", '', sent) for sent in data_re] # remove number

    data_words = list(sent_to_words(data_re))
    data_wo = remove_stopwords(data_words)

    #Topic Modeling
    id2word = corpora.Dictionary(data_wo)
    texts = data_wo
    corpus = [id2word.doc2bow(text) for text in texts]

    #Find optimize number of group
    dfs = {}
    for Ngroup in range(group_min, group_max+1):
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                       id2word=id2word,
                                                       num_topics=Ngroup,
                                                       random_state=100,
                                                       update_every=1,
                                                       chunksize=100,
                                                       passes=10,
                                                       alpha='auto',
                                                       per_word_topics=True)
        #create dataframe for Mapping topic with Text each Ngroup
        df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data)

        # Topic presentation
        vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
        pyLDAvis.save_html(vis, 'vis_{}_{}_topics.html'.format(save_as, Ngroup))

        # Format
        df_dominant_topic = df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
        df_dominant_topic['Request_Id'] = df_SWincident['Request ID']
        dfs[Ngroup] = df_dominant_topic

    # write mapping topic to excel
    writer = pd.ExcelWriter('{}.xlsx'.format(save_as))
    for sheet in range(group_min, group_max+1):
        dfs[sheet].to_excel(writer, sheet_name='{}_topics'.format(sheet))
    writer.save()
    writer.close()
