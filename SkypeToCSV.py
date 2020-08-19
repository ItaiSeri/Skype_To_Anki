import pandas as pd
import fasttext
import time
from pathlib import Path
from itertools import compress
from googletrans import Translator
from pycountry import languages


translator = Translator() #googletrans

#Local location of trained model for language detection. Explained in details here::
# https://fasttext.cc/docs/en/language-identification.html
fasttext_model_path=r"C:\Users\Itai\Anaconda3\Lib\site-packages\fasttext\lid.176.bin" 
fasttext_model = fasttext.load_model(fasttext_model_path)

src_lang='Chinese'
src_langgoogle_code= 'zh-CN' #code list can be found here: https://cloud.google.com/translate/docs/languages
start_date='2000-01-01'
end_date='2099-01-01'


def check_lang(character,src_lang):
    lang_list=[]
    predictions = fasttext_model.predict(character,k=5,threshold=0.7)[0]
    for pred in predictions:
        pred=pred.replace('__label__','')
        if languages.get(alpha_2=pred) is not None:
            lang_list.append(languages.get(alpha_2=pred).name)
    return src_lang in lang_list  

def SplitAndTrim(text,trim=True, split_del='\n'): 
    if trim == True:
        return text.replace(" ","").split(split_del)
    else:
        return text.split(split_del)



Skype_file_path=Path(r"F:\Dowloads\8_itai.seri_export\messages.json")
expath=Path("F:/Dowloads")
contact_list=['Anne Wu']
#'Anne Wu','Wendy K'

def Skype2Anki(Skype_file_path,contact_list,src_lang,trim=True, split_del='\n',start_date='2000-01-01',end_date='2099-01-01'):


    start = time.time()
    #Read conversations column from Skype's Json file, then normalize. I.E Break up Series to df with meaningful columns
    normalized=pd.json_normalize(pd.read_json(Skype_file_path,encoding='utf-8').conversations)
    
    #Convert the MessageList of every contact to df, then concat all of them into one df 
    df=pd.concat(normalized.MessageList[normalized.displayName.isin(contact_list)].apply(lambda x: pd.DataFrame(x)).tolist())
       
    #Convert originalarrivaltime to datetime & filter df by datetime
    df.originalarrivaltime=pd.to_datetime(df.originalarrivaltime)
    df=df[(df['originalarrivaltime'] > start_date) & (df['originalarrivaltime'] < end_date)]
    
    #delete white spaces needs to be only in asian languages
    df['content_trim_split']= df['content'].apply(lambda x: SplitAndTrim(x,trim, split_del))
     #apply the function for every word
    df['is_src_lang']= df['content_trim_split'].apply(lambda x: [check_lang(string,src_lang) for string in x])
    #Delete lists where text is not in the source language
    df['clean_content']= df[['content_trim_split','is_src_lang']].apply(tuple,axis=1).apply(lambda x: list(compress(x[0],x[1])))
    
    # Leave only rows with non empty list. "not x": True for empty list, False for not empty list
    # Split rows with multiple lists to multiple rows (explode)
    fltrd_df=df.clean_content[df['clean_content'].apply(lambda x: not x) ==False].explode().reset_index(drop=True) 
    
    #Transle every row to destination language
    Translation=fltrd_df.apply(lambda x: translator.translate(x).text)
    #Get pronunciation of source language for every row
    Pinyin=fltrd_df.apply(lambda x: translator.translate(x,dest=src_langgoogle_code).pronunciation)
    Anki_CSV=pd.concat([fltrd_df,Translation,Pinyin],axis=1)
    Anki_CSV.columns=['Source','Translation','Pinyin']
    file=expath/'Anki_CSV.txt'
    file.unlink(missing_ok=True) #Delete file if exists
    Anki_CSV.to_csv(file, index=None, mode='a')
    end = time.time()
    runtime= end - start
    print('CSV file created after '+ str(runtime) + ' seconds.\nLocation: ' +str(expath/'Anki_CSV.txt'))



    










