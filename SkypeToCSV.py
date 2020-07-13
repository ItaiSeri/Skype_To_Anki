import pandas as pd
import numpy as np
from pathlib import Path
from pandas.io.json import json_normalize

import googletrans
from googletrans import Translator

path=Path("F:\Dowloads\8_itai.seri_export/messages.json")
expath=Path("F:/Dowloads")

msg =pd.read_json(path) #output: DataFrame, all relavent data in 'conversations' column 
conver=msg.conversations #output: Series
norm=json_normalize(conver) #output: DataFrame, breaks up Series to DataFrame (with meaningfull columns)
########
an=norm.MessageList[norm.displayName=='Anne Wu']
an_df=pd.DataFrame(an[9])
#an_df.to_csv(expath/'conv.txt', header=None, index=None, mode='a')
############

multiple_df=norm.MessageList.apply(lambda x: pd.DataFrame(x))
df=pd.concat(multiple_df.tolist())
df.content.to_csv(expath/'conv.txt', header=None, index=None, mode='a')

translator = Translator()
print (translator.translate('小伙子怎么了',src='zh-CN').text)
print (translator.translate('hello world').src)