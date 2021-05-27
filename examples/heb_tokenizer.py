from yap_wrapper import HebTokenizer


if __name__=='__main__':
   import sys
   from itertools import islice

   from optparse import OptionParser
   parser = OptionParser("%prog [options] < in_file > out_file")
   parser.add_option("-i","--ie",help="input encoding [default %default]",dest="in_enc",default="utf_8_sig")
   parser.add_option("-o","--oe",help="output encoding [default %default]",dest="out_enc",default="utf_8")
   opts, args = parser.parse_args()

   #FILTER = set(['JUNK','ENG'])
   FILTER = set()

   #for sent in codecs.getreader(opts.in_enc)(sys.stdin):
      #print u"\n".join(["%s %s" % (which,tok) for which,tok in tokenize(sent) if which not in FILTER]).encode("utf8")
   sent="שלום לכם מלאכים קטנים. האם QUEEN היא הלהקה הטובה לשנת 1978?"
   print(sent)
   #print (' '.join([tok for (which,tok) in tokenize(sent)]).encode(opts.out_enc))

   ht=HebTokenizer()
   parts=ht.tokenize(sent)
   print(type(parts))
   for t in parts:
       print(t)