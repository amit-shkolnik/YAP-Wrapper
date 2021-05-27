from yap_wrapper import YapApi


if __name__ == '__main__':
    # The text to be processed.
    text = "עכשיו אני מרגיש כאילו לא יודע כלום עכשיו אני מחיש את צעדיי היא מסתכלת בחלון רואה אותי עובר בחוץ היא לא יודעת מה עובר עליי. \
    בתוך עיניה הכחולות ירח חם תלוי, עכשיו היא עצובה כמוני בוודאי היא מוציאה את בגדיי אוכלת לבדה ת'תות \
    היא לא יודעת, מה עובר עליי. \
    אמנם אין אהבה שאין לה סוף אבל הסוף הזה נראה לי מקולל הולך בין האנשים ברחוב צועק או או או או או או \
    תגידו לה."   
    
    # IP of YAP server, if locally installed then '127.0.0.1'
    ip='127.0.0.1:8000'
    yap=YapApi()    
    tokenized_text, segmented_text, lemmas, dep_tree, md_lattice, ma_lattice=yap.run(text, ip)                 
    print(tokenized_text)
    print(segmented_text)
    print(lemmas)
    print(dep_tree.to_string())
    print(md_lattice)
    print(ma_lattice)
    print('Program end')
        
    


