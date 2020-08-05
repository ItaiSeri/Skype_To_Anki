# coding: utf8

#import os, sys

import pandas as pd
import numpy as np
from pathlib import Path
from pandas.io.json import json_normalize
from itertools import compress

import googletrans
from googletrans import Translator
translator = Translator()

from pycountry import languages
import fasttext

fasttext_model_path=r"C:\Users\Itai\Anaconda3\Lib\site-packages\fasttext/lid.176.bin"
model = fasttext.load_model(fasttext_model_path)
src_lang_='Chinese'

sentences = 'wǒ qù dà chéng shì de diàn yǐng yuàn kàn diàn yǐng 。 \n我 去 大 城    市  的 电   影   院   看  电   影   。 '
def check_lang(character,src_lang_):
    lang_list=[]
    predictions = model.predict(character,k=5,threshold=0.7)[0]
    for pred in predictions:
        pred=pred.replace('__label__','')
        if languages.get(alpha_2=pred) is not None:
            lang_list.append(languages.get(alpha_2=pred).name)
    return src_lang_ in lang_list  


src_lang='zh-CN'

path=Path("F:\Dowloads\8_itai.seri_export/messages.json")
expath=Path("F:/Dowloads")

msg =pd.read_json(path,encoding='utf-8') #output: DataFrame, all relavent data in 'conversations' column 
conver=msg.conversations #output: Series
norm=json_normalize(conver) #output: DataFrame, breaks up Series to DataFrame (with meaningfull columns)
########
an=norm.MessageList[norm.displayName=='Anne Wu']
an_df=pd.DataFrame(an[9])
an_df.content.to_csv(expath/'conv.txt', header=None, index=None, mode='a')
an_df['content_trim_split']= an_df['content'].apply(lambda x: x.replace(" ","").splitlines()) #delete white spaces needs to be only in asian languages
#an_df['is_src_lang']= an_df['content'].apply(lambda x: [translator.detect(word).lang==src_lang for word in x.split()])
an_df['is_src_lang']= an_df['content_trim_split'].apply(lambda x: [check_lang(string,src_lang_) for string in x])
an_df['combine']= an_df[['content_trim_split','is_src_lang']].apply(tuple,axis=1)
an_df['clean_content']=an_df['combine'].apply(lambda x: list(compress(x[0],x[1])))
an_df['fltr']= an_df['clean_content'].apply(lambda x: not x)
fltrd_an_df=an_df.clean_content[an_df['fltr']==False].explode()
#an_df['tr']=an_df.content_trimmed.apply(lambda x: translator.translate(x).text)
############

multiple_df=norm.MessageList.apply(lambda x: pd.DataFrame(x))
df=pd.concat(multiple_df.tolist())
#df.content.to_csv(expath/'conv.txt', header=None, index=None, mode='a')

if not an_df.clean_content[0]:
    print ('empty')
    










