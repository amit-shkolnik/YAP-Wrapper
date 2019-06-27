# YAP-Wrapper
Python interface to Open University YAP (Yet another parser) https://github.com/OnlpLab/yap.

How to use:
========
1. Install YAP (instruction are on YAP page....)
2. After installing YAP, run it as HTTP server, by simply run "./yap api" from command line. Now Yap is running as HTTP serve on port 8000.
3. Run the Python code in yap_api.py file ==> main method.

  	3.1 If YAP is not installed locally, set the IP address on yap_api.py main method.
4. The code return 6 elements:
  tokenized_text- string. The text tokenized.
  segmented_text - string. The text segmented.
  lemmas - string. The lemmas of the text.
  dep_tree - DataFrame. Dependency tree.
  md_lattice - DataFrame. Morphological analysis, as decided by YAP
  ma_lattice - DataFrame. Morphological analysis inlcuding all possible lattices.

This code is fully free under Apaceh 2.0 (https://www.apache.org/licenses/LICENSE-2.0) (no guarantees!). 


