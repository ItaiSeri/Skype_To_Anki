# coding: utf8

#import os, sys

import pandas as pd
import numpy as np
from pathlib import Path
from pandas.io.json import json_normalize

import googletrans
from googletrans import Translator
translator = Translator()
import cld2


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
an_df['content_trimmed']= an_df['content'].str.replace(" ","")
#an_df['is_src_lang']= an_df['content'].apply(lambda x: [translator.detect(word).lang==src_lang for word in x.split()])
#an_df['tr']=an_df.content_trimmed.apply(lambda x: translator.translate(x).text)
############

multiple_df=norm.MessageList.apply(lambda x: pd.DataFrame(x))
df=pd.concat(multiple_df.tolist())
#df.content.to_csv(expath/'conv.txt', header=None, index=None, mode='a')

#isReliable, textBytesFound, details = cld2.detect("王明：那是杂志吗")
#print('  reliable: %s' % (isReliable != 0))
#print('  textBytes: %s' % textBytesFound)
#print('  details: %s' % str(details))

from pycountry import languages
import fasttext
fasttext_model_path=r"C:\Users\Itai\Anaconda3\Lib\site-packages\fasttext/lid.176.bin"
model = fasttext.load_model(fasttext_model_path)

sentences = ['需']
predictions = model.predict(sentences,k=5)[0][0]
for pred in predictions:
    print (languages.get(alpha_2=pred.replace('__label__','')).name)
#print(predictions)

lang_name = languages.get(alpha_2=model.predict(sentences)[0][0][0].replace('__label__','')).name
#print(lang_name)








