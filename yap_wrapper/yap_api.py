# Author: Amit Shkolnik
# Python Version: 3.6


## Copyright 2019 Amit Shkolnik
##
##    This program is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation, either version 3 of the License, or
##    (at your option) any later version.
##
##    This program is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import pandas as pd
import numpy as np
import requests
import sys
import traceback
import csv
import re, string
from .enums import *
from .hebtokenizer import HebTokenizer



class YapApi(object):
    """
    Interface to Open University YAP (Yet another parser) https://github.com/OnlpLab/yap.
    This class is calling GO baesd server, and:
    1. Wrap YAP in Python.
    2. Add tokenizer. Credit: Prof' Yoav Goldberg.
    3. Turn output CONLLU format to Dataframe & JSON.
    """   

    def __init__(self):        
        pass  
       
    def run(self, text:str, ip:str):
        """
        text: the text to be parsed.
        ip: YAP server IP, with port (default is 8000), if localy installed then 127.0.0.1:8000
        """
        try:
            print('Start Yap call')
            # Keep alpha-numeric and punctuations only.
            alnum_text=self.clean_text(text)
            # Tokenize...
            tokenized_text = HebTokenizer().tokenize(alnum_text)
            tokenized_text = ' '.join([word for (part, word) in  tokenized_text])
            print("Tokens: {}".format(len(tokenized_text.split())))                       
            self.init_data_items()                        
            # Split to sentences for best performance.
            text_arr=self.split_text_to_sentences(tokenized_text)
            for i, sntnce_or_prgrph in enumerate( text_arr):
                # Actual call to YAP server
                rspns=self.call_yap(sntnce_or_prgrph, ip)
                print('End Yap call {} /{}'.format( i ,len(text_arr)-1))  
                # Expose this code to print the results iin Conllu format
                #conllu_dict=self.print_in_conllu_format(rspns)
                # Turn CONLLU format to dataframe
                _dep_tree, _md_lattice, _ma_lattice=self.conllu_format_to_dataframe(rspns)
                _segmented_text= ' '.join( _dep_tree[yap_cols.word.name])
                _lemmas=' '.join(_dep_tree[yap_cols.lemma.name])
                self.append_prgrph_rslts(_dep_tree, _md_lattice, _ma_lattice, _segmented_text, _lemmas)                 
            return tokenized_text, self.segmented_text, self.lemmas, self.dep_tree, self.md_lattice, self.ma_lattice            

        except Exception as err:
            print( sys.exc_info()[0])            
            print( traceback.format_exc())          
            print( str(err))    
        print("Unexpected end of program")
        
    def split_text_to_sentences(self, tokenized_text):
        """
        YAP better perform on sentence-by-sentence.
        Also, dep_tree is limited to 256 nodes.
        """
        max_len=150
        arr=tokenized_text.strip().split()        
        sentences=[]
        # Finding next sentence break.        
        while (True):
            stop_points=[h for h in [i for i, e in enumerate(arr) if re.match(r"[!|.|?]",e)] ]
            if len(stop_points)>0:
                stop_point=min(stop_points)
                # Keep several sentence breaker as 1 word, like "...." or "???!!!"
                while True:
                    stop_points.remove(stop_point)
                    if len(stop_points)>1 and min(stop_points)==(stop_point+1):
                        stop_point=stop_point+1
                    else:
                        break
                # Case there is no sentence break, and this split > MAX LEN:
                sntnc=arr[:stop_point+1]
                if len(sntnc) >max_len:
                    while(len(sntnc) >max_len):
                        sentences.append(" ".join(sntnc[:140]))
                        sntnc=sntnc[140:]
                    sentences.append(" ".join(sntnc))
                # Normal: sentence is less then 150 words...
                else: 
                    sentences.append(" ".join(arr[:stop_point+1] ))
                arr=arr[stop_point+1:]                                 
            else:
                break       
        if len(arr)>0:            
            sentences.append(" ".join(arr))
        return sentences
          
    def clean_text(self, text:str):
        text=text.replace('\n', ' ').replace('\r', ' ')        
        pattern= re.compile(r'[^א-ת\s.,!?a-zA-Z]')        
        alnum_text =pattern.sub(' ', text)
        while(alnum_text.find('  ')>-1):
            alnum_text=alnum_text.replace('  ', ' ')
        return alnum_text            

    def init_data_items(self):
        self.segmented_text=""
        self.lemmas=""
        self.dep_tree=pd.DataFrame()
        self.md_lattice=pd.DataFrame()
        self.ma_lattice=pd.DataFrame()            

    def append_prgrph_rslts(self, _dep_tree:pd.DataFrame, _md_lattice:pd.DataFrame, _ma_lattice:pd.DataFrame, 
                            _segmented_text:str, _lemmas:str):  
        self.segmented_text="{} {}".format(self.segmented_text, _segmented_text).strip()
        self.lemmas="{} {}".format(self.lemmas, _lemmas).strip()
        self.dep_tree=pd.concat([self.dep_tree, _dep_tree])
        self.md_lattice=pd.concat([self.md_lattice, _md_lattice])
        self.ma_lattice=pd.concat([self.ma_lattice, _ma_lattice])

    def split_long_text(self, tokenized_text:str):
        # Max num of words YAP can handle at one call.
        max_len=150
        arr=tokenized_text.split()        
        rslt=[]
        while len(arr)> max_len:          
            # Finding next sentence break.
            try:                
                stop_point=min([h for h in [i for i, e in enumerate(arr) if re.match(r"[!|.|?]",e)] if h> max_len])                            
            except Exception as err:
                if str(err) =="min() arg is an empty sequence":
                    stop_point=150
                if len(arr)<stop_point:
                    stop_point=len(arr)-1
            rslt.append(" ".join(arr[: (stop_point+1)]))
            arr=arr[(stop_point+1):]
        rslt.append(" ".join(arr))
        return rslt    

    def call_yap(self, text:str, ip:str):
        """
        Actual call to YAP HTTP Server
        """
        url = "{}{}{}".format( "http://", ip, "/yap/heb/joint")
        _json='{"text":"  '+text+'  "}'         
        headers = {'content-type': 'application/json'}
        r = requests.post(url,
                  data=_json.encode('utf-8'),
                  headers={'Content-type': 'application/json; charset=utf-8'})
        self.check_response_status(r)
        return r.json()

    def check_response_status(self, response: requests.Response):
         if response.status_code != 200:
                print('url: %s' %(response.url))                
                if response.json() != None:
                    print("response : {}".format( response.json()))                         
                if response.text != None:
                    print('Reason: Text: %s'%( response.text))

    def conllu_format_to_dataframe(self, rspns:dict):       
        for k,v in rspns.items():           
            if k==yap_ent.dep_tree.name:
                dep_tree=self.parse_dep_tree(v)
            elif k==yap_ent.md_lattice.name:
                md_lattice=self.parse_md_lattice(v)
            elif k==yap_ent.ma_lattice.name:
                ma_lattice=self.parse_ma_lattice(v)
        return dep_tree.fillna(-1), md_lattice.fillna(-1), ma_lattice.fillna(-1)

    def parse_dep_tree(self, v:str):
        data=[sub.split("\t") for item  in str(v).split("\n\n") for sub in item.split("\n")]
        labels=[yap_cols.num.name, yap_cols.word.name, yap_cols.lemma.name, yap_cols.pos.name, yap_cols.pos_2.name, 
                yap_cols.empty.name, yap_cols.dependency_arc.name, yap_cols.dependency_part.name,
                        yap_cols.dependency_arc_2.name, yap_cols.dependency_part_2.name]        
        # remove first line w=hich is empty
        data=[l for l in data if len(l)!=1 ]
        # remove stop char 
        new_data = []
        for row in data:
            n_row = [word.replace("\r","") for word in row]
            new_data.append(n_row)        
        df=pd.DataFrame.from_records(new_data, columns=labels)
        # Case YAP find punctuation chars like ',', '.', YAP set no lemma for them.
        # That case set the word to be its own lemma
        df.loc[df[yap_cols.lemma.name] == '', [yap_cols.lemma.name]] =df[yap_cols.word.name]
        return df

    def parse_md_lattice(self, v:str):
        data=[sub.split("\t") for item  in str(v).split("\n\n") for sub in item.split("\n")]
        labels=[yap_cols.num.name, yap_cols.num_2.name,yap_cols.word.name, yap_cols.lemma.name, 
                yap_cols.pos.name, yap_cols.pos_2.name,                
                # parts:
                yap_cols.gen.name, 
                yap_cols.num_last.name, 
                yap_cols.num_s_p.name, yap_cols.per.name, yap_cols.tense.name ]         

        list_of_dict=[]
        for row in data: 
            if len(row)==1:
                continue
            if len(row)!=8:
                raise Exception("Len of row is: {} row: {}".format(len(row), row))
            _dict={
                yap_cols.num.name:None, 
                yap_cols.num_2.name:None,
                yap_cols.word.name:None, 
                yap_cols.empty.name:None, 
                yap_cols.pos.name:None, 
                yap_cols.pos_2.name:None,
                yap_cols.gen.name:None, 
                yap_cols.num_s_p.name:None, 
                yap_cols.per.name:None, 
                yap_cols.tense.name:None, 
                yap_cols.num_last.name:None                       
                }                              
            for i,tag in enumerate( row):               
                if i<6 or i==7:
                    _dict[labels[i]]=tag 
                else:
                    for part in tag.split("|"):
                        if part.split("=")[0]==yap_cols.gen.name:    
                            _dict[yap_cols.gen.name]=part.split("=")[1]
                        elif part.split("=")[0]==yap_cols.per.name:    
                            _dict[yap_cols.per.name]=part.split("=")[1]
                        elif part.split("=")[0]==yap_cols.tense.name:    
                            _dict[yap_cols.tense.name]=part.split("=")[1] 
                        elif part.split("=")[0]==yap_cols.num.name:    
                            _dict[yap_cols.num_s_p.name]=part.split("=")[1]
            list_of_dict.append(_dict)
        df=pd.DataFrame(list_of_dict)
        return df

    def parse_ma_lattice(self, v:str):
        data=[sub.split("\t") for item  in str(v).split("\n\n") for sub in item.split("\n")]
        labels=[yap_cols.num.name, 
                yap_cols.num_2.name,
                yap_cols.word.name, 
                yap_cols.lemma.name,
                yap_cols.empty.name, 
                yap_cols.pos.name, 
                yap_cols.pos_2.name,                
                # parts:
                yap_cols.gen.name, 
                # Should remain on #7 position.
                yap_cols.num_last.name, 
                yap_cols.num_s_p.name, yap_cols.per.name, 
                yap_cols.tense.name,                 
                yap_cols.suf_gen.name,
                yap_cols.suf_num.name,
                yap_cols.suf_per.name
                ]         
        list_of_dict=[]
        for row in data: 
            if len(row)==1:
                continue
            if len(row)!=8:
                raise Exception("Len of row is: {} row: {}".format(len(row), row))
            _dict={
                yap_cols.num.name:None, 
                yap_cols.num_2.name:None,
                yap_cols.word.name:None, 
                yap_cols.lemma.name:None,
                yap_cols.empty.name:None, 
                yap_cols.pos.name:None, 
                yap_cols.pos_2.name:None,
                yap_cols.gen.name:None, 
                yap_cols.num_s_p.name:None, 
                yap_cols.per.name:None, 
                yap_cols.tense.name:None, 
                yap_cols.num_last.name:None,
                yap_cols.suf_gen.name:None,
                yap_cols.suf_num.name:None,
                yap_cols.suf_per.name:None                       
                }                              
            for i,tag in enumerate( row):
                if i<6 or i==7:
                    _dict[labels[i]]=tag
                else:
                    for part in tag.split("|"):
                        if part.split("=")[0]==yap_cols.gen.name:    
                            _dict[yap_cols.gen.name]=part.split("=")[1]
                        elif part.split("=")[0]==yap_cols.per.name:    
                            _dict[yap_cols.per.name]=part.split("=")[1]
                        elif part.split("=")[0]==yap_cols.tense.name:    
                            _dict[yap_cols.tense.name]=part.split("=")[1] 
                        elif part.split("=")[0]==yap_cols.num.name:    
                            _dict[yap_cols.num_s_p.name]=part.split("=")[1]                        
                        elif part.split("=")[0]==yap_cols.suf_gen.name:    
                            _dict[yap_cols.suf_gen.name]=part.split("=")[1]
                        elif part.split("=")[0]==yap_cols.suf_num.name:    
                            _dict[yap_cols.suf_num.name]=part.split("=")[1]
                        elif part.split("=")[0]==yap_cols.suf_per.name:    
                            _dict[yap_cols.suf_per.name]=part.split("=")[1]
            list_of_dict.append(_dict)
        df=pd.DataFrame(list_of_dict)
        return df            

    def print_in_conllu_format(self, rspns:dict):
       new_dict={}
       for k,v in rspns.items():           
           new_dict[k]=[]
           print("")
           print(k)
           for item in str( v).split("\n\n"):
               for sub_item in item.split("\n"):               
                    if sub_item!="":
                        print(sub_item)
                        new_dict[k].append(sub_item)

