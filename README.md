# YAP-Wrapper
Python interface to Open University YAP (Yet another parser) https://github.com/OnlpLab/yap.

How to use:
========
1. Install YAP (instruction are on YAP page....)
2. After installing YAP, run it as HTTP server, by simply run "./yap api" from command line. Now Yap is running as HTTP serve on port 8000.
3. Run the Python code in yap_api.py file ==> main method.
The python code is calling YAP server via HTTP request...

  	3.1 If YAP is not installed locally, set the IP address on yap_api.py main method.
4. The code return 6 elements:
* tokenized_text- string. The text tokenized.
* segmented_text - string. The text segmented.
* lemmas - string. The lemmas of the text.
* dep_tree - DataFrame. Dependency tree.
* md_lattice - DataFrame. Morphological analysis, as decided by YAP.
* ma_lattice - DataFrame. Morphological analysis inlcuding all possible lattices.

For example, if original text is: "בתוך עיניה הכחולות"
Then output is:
* tokenized_text- "בתוך עיניה הכחולות" 
* segmented_text - "בתוך עיניה ה כחולות"
* lemmas - "בתוך עין ה כחול"
* dep_tree - 

![alt text](https://github.com/amit-shkolnik/YAP-Wrapper/blob/master/dep_tree.png)
* md_lattice -

![alt text](https://github.com/amit-shkolnik/YAP-Wrapper/blob/master/md_lattice.png)
* ma_lattice - 

![alt text](https://github.com/amit-shkolnik/YAP-Wrapper/blob/master/ma_lattice.png)


This code is fully free under Apaceh 2.0 (https://www.apache.org/licenses/LICENSE-2.0) (no guarantees!). 


