# coding: utf8

#import os, sys

import pandas as pd
import numpy as np
from pathlib import Path
from pandas.io.json import json_normalize

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
    predictions = model.predict(character,k=5)[0]
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
an_df['content_trim_split']= an_df['content'].str.replace(" ","").splitlines()
#an_df['is_src_lang']= an_df['content'].apply(lambda x: [translator.detect(word).lang==src_lang for word in x.split()])
an_df['is_src_lang']= an_df['content'].apply(lambda x: [check_lang(word,src_lang_) for word in x.split()])
#an_df['tr']=an_df.content_trimmed.apply(lambda x: translator.translate(x).text)
############

multiple_df=norm.MessageList.apply(lambda x: pd.DataFrame(x))
df=pd.concat(multiple_df.tolist())
#df.content.to_csv(expath/'conv.txt', header=None, index=None, mode='a')


for x in b:
    print (check_lang(x,src_lang_))

 #check_lang(x,src_lang_ 
#print(predictions)

#if languages.get(alpha_2='he') is not None:
#    lang_name = languages.get(alpha_2='he').name
#print(lang_name)









