from enum import Enum 

class PennTags(Enum):
    CC   =1                        #Coordinating conjunction 
    CD   = 2                       #Cardinal number 
    DT   =  3                      #Determiner 
    EX   =   4                     #Existential there 
    FW   =  5                      #Foreign word 
    IN   =  6                      #Preposition or subordinating conjunction 
    JJ   =  7                      #Adjective 
    JJR   = 8                       #Adjective, comparative 
    JJS   =  9                      #Adjective, superlative 
    LS   =  10                      #List item marker 
    MD   =  11                      #Modal 
    NN   =  12                      #Noun, singular or mass 
    NNS   = 13                       #Noun, plural 
    NNP   = 14                       #Proper noun, singular     
    NNPS   = 15                       #Proper noun, plural 
    PDT   =  16                      #Predeterminer 
    POS   =  17                      #Possessive ending 
    PRP   = 18                       #Personal pronoun 
    #PRP$   = 19                        #Possessive pronoun 
    RB   = 20                       #Adverb 
    RBR   = 21                        #Adverb, comparative 
    RBS   = 22                       #Adverb, superlative 
    RP   = 23                       #Particle 
    SYM   = 24                       #Symbol 
    TO   = 25                       #to 
    UH   = 26                       #Interjection 
    VB   = 27                       #Verb, base form 
    VBD   = 28                       #Verb, past tense 
    VBG   = 29                       #Verb, gerund or present participle 
    VBN   = 30                       #Verb, past participle 
    VBP   = 31                       #Verb, non-3rd person singular present 
    VBZ   = 32                       #Verb, 3rd person singular present 
    WDT   = 33                       #Wh-determiner 
    WP   =  34                      #Wh-pronoun 
    #WP$   = 35                       #Possessive wh-pronoun 
    WRB   =  36                      #Wh-adverb

    # YAP SPECIFIC TAGS
    BN=37
    BNT=38
    NNT  =39                 

class yap_cols(Enum):
    num=1
    num_2=2
    word=3
    pos=4
    pos_2=5
    dependency_arc=6
    dependency_part=7
    gen=8   # gender
    num_s_p=9
    tense=10 # past, Imperative
    suf_gen =110 # suffix gender?
    suf_per=12 
    suf_num=13
    num_last=14
    dependency_arc_2=15
    dependency_part_2=16
    other=17
    empty=18
    per=19
    empty_2=20
    empty_3=21
    stemmed=22
    verified=23
    lemma=24

class yap_ent(Enum):
    tokenized_text=1
    segmented_text=2
    lemmas=3
    dep_tree=4
    md_lattice=5
    ma_lattice=6
    

class app_enum(object):
     segmented_text="segmented_text"
     stemmed_str="stemmed_str"
