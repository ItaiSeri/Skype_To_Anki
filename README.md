# Skype_To_Anki
Skype is a very common tool among language learners.
The aim of this project is coverting foriegn language text found in Skype's chat history into a CSV file that can be used as flashcards in Anki.
Your Skype chat history can be found  [here](https://support.skype.com/en/faq/FA34894/how-do-i-export-my-skype-files-and-chat-history).

### Prerequisites
To install, either use things like pip with the package name or download the package and put the directory into your python path.
Packages in use: `pandas, fasttext, time, pathlib, itertools, googletrans, pycountry`.

In addition, you will need to download a file that contains a model for language identification. 
Download from [here](https://fasttext.cc/docs/en/language-identification.html) the file named: *lid.176.bin*
