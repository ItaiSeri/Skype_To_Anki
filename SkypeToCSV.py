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


src_lang='zh-CN'

Skype_Json=Path(r"F:\Dowloads\8_itai.seri_export\messages.json")
expath=Path("F:/Dowloads")
contact_list=['Wendy K','Anne Wu']
#def Skype2Anki(contact_list):
    #add date slicing - try finding something more accurate than original arrival time
    #maby filter with messagetype
    
#Read conversations column from Skype's Json file, then normalize. I.E Break up Series to df with meaningful columns
normalized=json_normalize(pd.read_json(Skype_Json,encoding='utf-8').conversations)

#Convert the MessageList of every contact to df, then concat all of them into one df 
df=pd.concat(normalized.MessageList[normalized.displayName.isin(contact_list)].apply(lambda x: pd.DataFrame(x)).tolist())

df['content_trim_split']= df['content'].apply(lambda x: x.replace(" ","").splitlines()) #delete white spaces needs to be only in asian languages
df['is_src_lang']= df['content_trim_split'].apply(lambda x: [check_lang(string,src_lang_) for string in x])
df['combine']= df[['content_trim_split','is_src_lang']].apply(tuple,axis=1)
df['clean_content']=df['combine'].apply(lambda x: list(compress(x[0],x[1])))
df['fltr']= df['clean_content'].apply(lambda x: not x)
fltrd_df=df.clean_content[df['fltr']==False].explode().reset_index(drop=True)
Translation=fltrd_df.apply(lambda x: translator.translate(x).text)
Pinyin=fltrd_df.apply(lambda x: translator.translate(x,dest='zh-CN').pronunciation)
Anki_CSV=pd.concat([fltrd_df,Translation,Pinyin],axis=1)
Anki_CSV.columns=['Source','Translation','Pinyin']
Anki_CSV.to_csv(expath/'Anki_CSV.txt', index=None, mode='a')


    










