#!/usr/bin/env python
# encoding- utf8

## Copyright 2010 Yoav Goldberg
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

import re
import codecs
import math
"""
    Robust Hebrew Tokenizer 

    works as a filter:
    hebtokenizer.py < in > out
   
    run as:
    hebtokenizer.py -h  
    for options

    KNOWN ISSUES:
        - NOT VERY FAST!!!

        - transition from hebrew words to numbers: ב-23:00  will be cut as ב-23 :00
        - deliberately not segmenting משהוכלב from start of words before numbers/quotes/dashes
        - emoticons are not supported (treated as punctuation)
        - ' is always kept at end of hebrew chunks (a document level pre/post processing could help here)
        - !!!!!!!111111 are split to !!!!!!!! 1111111
"""
#########

def heb(s,t): return ('HEB',t)
def eng(s,t): return ('ENG',t)
def num(s,t): return ('NUM',t)
def url(s,t): return ('URL',t)
def punct(s,t): return ('PUNCT',t)
def junk(s,t): return ('JUNK',t)

#### patterns
_NIKUD = u"\u05b0-\u05c4"
_TEAMIM= u"\u0591-\u05af"

undigraph = lambda x:x.replace(u"\u05f0",u"וו").replace(u"\u05f1",u"וי").replace("\u05f2","יי").replace("\ufb4f","אל").replace(u"\u200d","")

_heb_letter = r"([א-ת{}]|[דגזצתט]')".format( _NIKUD)

# a heb word including single quotes, dots and dashes  / this leaves last-dash out of the word
_heb_word_plus = r"[א-ת%s]([.'`\"\-/\\]?['`]?[א-ת%s0-9'`])*" % (_NIKUD,_NIKUD)

# english/latin words  (do not care about abbreviations vs. eos for english)
_eng_word = r"[a-zA-Z][a-zA-Z0-9'.]*"  

# numerical expression (numbers and various separators)
#_numeric = r"[+-]?[0-9.,/\-:]*[0-9%]"
_numeric = r"[+-]?([0-9][0-9.,/\-:]*)?[0-9]%?"

# url
_url = r"[a-z]+://\S+"

# punctuations
_opening_punc = r"[\[('`\"{]"
_closing_punc = r"[\])'`\"}]"
_eos_punct = r"[!?.]+"
_internal_punct = r"[,;:\-&]"

# junk
#_junk = ur"[^א-ת%sa-zA-Z0-9%%&!?.,;:\-()\[\]{}\"'\/\\+]+" #% _NIKUD
_junk = r"[^א-ת%sa-zA-Z0-9!?.,:;\-()\[\]{}]+" % _NIKUD #%%&!?.,;:\-()\[\]{}\"'\/\\+]+" #% _NIKUD

is_all_heb = re.compile(r"^%s+$" % (_heb_letter),re.UNICODE).match
is_a_number = re.compile(r"^%s$" % _numeric ,re.UNICODE).match
is_all_lat= re.compile(r"^[a-zA-Z]+$",re.UNICODE).match
is_sep = re.compile(r"^\|+$").match
is_punct = re.compile(r"^[.?!]+").match




#### scanner

scanner = re.Scanner([
    (r"\s+", None),
    (_url, url),
    (_heb_word_plus, heb),
    (_eng_word, eng),
    (_numeric,  num),
    (_opening_punc, punct),
    (_closing_punc, punct),
    (_eos_punct, punct),
    (_internal_punct, punct),
    (_junk, junk),
    ])

##### tokenize
def tokenize( sent):
    tok = sent
    parts,reminder = scanner.scan(tok)
    assert(not reminder)
    return parts


class HebTokenizer(object):
    def tokenize(self, sent:str):
        if type(sent)!= str:
            return []
        if sent is None or len(sent)==0 :
            return []
        return tokenize(sent)
