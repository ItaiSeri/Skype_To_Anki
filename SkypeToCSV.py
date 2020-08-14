import pandas as pd
import fasttext
from pathlib import Path
from pandas.io.json import json_normalize
from itertools import compress
from googletrans import Translator
from pycountry import languages


translator = Translator() #googletrans

#Local location of trained model for language detection. Explained in details here::
# https://fasttext.cc/docs/en/language-identification.html
fasttext_model_path=r"C:\Users\Itai\Anaconda3\Lib\site-packages\fasttext\lid.176.bin" 
fasttext_model = fasttext.load_model(fasttext_model_path)

src_lang_='Chinese'


def check_lang(character,src_lang_):
    lang_list=[]
    predictions = fasttext_model.predict(character,k=5,threshold=0.7)[0]
    for pred in predictions:
        pred=pred.replace('__label__','')
        if languages.get(alpha_2=pred) is not None:
            lang_list.append(languages.get(alpha_2=pred).name)
    return src_lang_ in lang_list  

def SplitAndTrim(x): 
    return x.replace(" ","").splitlines()
    
src_lang='zh-CN'

Skype_Json=Path(r"F:\Dowloads\8_itai.seri_export\messages.json")
expath=Path("F:/Dowloads")
contact_list=['Wendy K','Anne Wu']
#def Skype2Anki(contact_list):
    #add date slicing - try finding something more accurate than original arrival time
    #maby filter with messagetype
    #add .split('') to split by for instance =. like in Wendy K
    
#Read conversations column from Skype's Json file, then normalize. I.E Break up Series to df with meaningful columns
normalized=json_normalize(pd.read_json(Skype_Json,encoding='utf-8').conversations)

#Convert the MessageList of every contact to df, then concat all of them into one df 
df=pd.concat(normalized.MessageList[normalized.displayName.isin(contact_list)].apply(lambda x: pd.DataFrame(x)).tolist())

#delete white spaces needs to be only in asian languages
df['content_trim_split']= df['content'].apply(SplitAndTrim)
 #apply the function for every word
df['is_src_lang']= df['content_trim_split'].apply(lambda x: [check_lang(string,src_lang_) for string in x])
#Delete lists where text is not in the source language
df['clean_content']= df[['content_trim_split','is_src_lang']].apply(tuple,axis=1).apply(lambda x: list(compress(x[0],x[1])))

# Leave only rows with non empty list. "not x": True for empty list, False for not empty list
# Split rows with multiple lists to multiple rows (explode)
fltrd_df=df.clean_content[df['clean_content'].apply(lambda x: not x) ==False].explode().reset_index(drop=True) 

#Transle every row to destination language
Translation=fltrd_df.apply(lambda x: translator.translate(x).text)
#Get pronunciation of source language for every row
Pinyin=fltrd_df.apply(lambda x: translator.translate(x,dest='zh-CN').pronunciation)
Anki_CSV=pd.concat([fltrd_df,Translation,Pinyin],axis=1)
Anki_CSV.columns=['Source','Translation','Pinyin']
Anki_CSV.to_csv(expath/'Anki_CSV.txt', index=None, mode='a')


    










