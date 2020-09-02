
# -*- coding: utf-8 -*-

import sys
import random
import json
import io
import unicodecsv as csv
import difflib
import numpy as np
import mm_segmenter as mm
import re
import os
from collections import OrderedDict
from mm_converter import zawgyi_to_unicode
import json
import codecs

class DataProcessing:
    def __init__(self, *args, **kwargs):
        self.segmenter = mm.Segmenter()
    def process_annotated_data(self, src_file, target_file, analysis_file):
        non_burmese_pattern = re.compile("[^"
                                        u"\U00001000-\U0000109F"
                                        u"\U00000020"
                                        "]+", flags=re.UNICODE)
        
        with open(src_file, "r") as original_annotated_data:
            json_data = json.load(original_annotated_data)
            # raw_text = original_annotated_data.read()
            # unicode_arr = []
            for i, d_obj in enumerate(json_data):
                json_data[i]["content"] = zawgyi_to_unicode(d_obj["content"])
                json_data[i]["content"] = " ".join(self.segmenter.segment(json_data[i]["content"]))
                json_data[i]["content"] = re.sub("\s+", ' ', non_burmese_pattern.sub('', json_data[i]["content"]))
            with codecs.open("json_segmented_converted_cleaned.json", "w", encoding="utf-8") as out:
                json.dump(json_data, out, ensure_ascii=False, indent=4)
            # print(len(raw_text))
            # ans = []
            # for line in raw_text:
                # cleaned_text = non_burmese_pattern.sub('', line)
                # no_extra_spaces = re.sub("[\s\s+]", ' ', cleaned_text)
                # ans.append(no_extra_spaces)
            

            # Uncomment this part of the code for segmentation.
            # org_tokens = raw_text.split(" ")
            # num_org_tokens = len(org_tokens)
            # print(num_org_tokens)
            # raw_text_arr = []
            # count = 0
            # final_arr = []
            # tokens = 0
            # while count < len(org_tokens)+1000:
                # s_tring = " ".join(org_tokens[count:min(count+1000, len(org_tokens))])
                # raw_text_arr.append(s_tring)
                # count += 1000
                # segmented_arr = self.segmenter.segment(s_tring)
                # tokens += len(segmented_arr)
                # segmented_text = " ".join(segmented_arr)
                # final_arr.append(segmented_text)
                # print(segmented_text)
                # print(count)
            # final_arr = [] 
            # for i,line in enumerate(unicode_arr):
                
                # segmented_arr = self.segmenter.segment(line)
                # # tokens += len(segmented_arr)
                # segmented_text = " ".join(segmented_arr)
                # final_arr.append(segmented_text)
                # if i%50 == 0:
                    # print(i)
                    # final_text = "\n".join(final_arr)
                    # with open(target_file, "a+", encoding="utf-8") as output_file:
                        # output_file.write(final_text)
                    # final_arr = []
                    # final_text = ""



            # final_text = "\n".join(final_arr)
            # print(segmented_text)

            # with open(analysis_file, "w") as analysis:
                # analysis.write("UDHR Dump: \n Num tokens before segmentation: 1252 \n Num tokens after segmentation: {}".format(tokens))


            
            # with open(target_file, "w", encoding="utf-8") as output_file:
                # for line in ans:
                    # output_file.write(line+"\n")
    

if __name__ == "__main__":
    dataProcessor = DataProcessing()
    project_dir = "/home/harshil/Harshil/gt/spring2020/research2/burmese-NLP/data-collection/code/data_processing/data/"
    # src_file = "data/burmese_wiki/wiki_data_1.json"
    # target_file = "data/burmese_wiki/wiki_data_1_processed.json"
    # src_file = "data/wiki_dump/mywiki-20190201-pages-meta-current_processed.txt"
    # target_file = "data/wiki_dump/mywiki-20190201-pages-meta-current_processed.1.txt"
    # src_file = "data/final_data/unsegmented/udhr-unsegmented.1.txt" 
    # target_file = "data/final_data/segmented/udhr-segmented.txt"
    # analysis_file = "data/final_data/analysis/udhr_analysis.txt"
    src_file = "final_hs_data.json"
    target_file = ""
    analysis_file = "" 
    dataProcessor.process_annotated_data(os.path.join(
        project_dir+src_file), os.path.join(project_dir+target_file), os.path.join(project_dir+analysis_file))
