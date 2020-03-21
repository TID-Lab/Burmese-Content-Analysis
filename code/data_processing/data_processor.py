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


class DataProcessing:
    def __init__(self, *args, **kwargs):
        self.segmenter = mm.Segmenter()

    # removes all ascii characters.
    def process_annotated_data(self, src_file, target_file):
        non_burmese_pattern = re.compile("[^"
                                         u"\U00001000-\U0000109F"
                                         u"\U00000020"
                                         "]+", flags=re.UNICODE)
        with open(src_file, "r", encoding="utf-8") as original_annotated_data:
            original_text = OrderedDict(json.load(original_annotated_data))

            for key, val in original_text.items():
                raw_text = val["text-data"]
                cleaned_text = non_burmese_pattern.sub('', raw_text)
                segmented_text = self.segmenter.segment(cleaned_text)
                original_text[key]["token-count"] = len(segmented_text)
                original_text[key]["text-data"] = cleaned_text
                original_text[key]["segmented-data"] = segmented_text

            with open(target_file, "w", encoding="utf-8") as output_file:
                json.dump(original_text, output_file,
                          indent=4, ensure_ascii=False)


if __name__ == "__main__":
    dataProcessor = DataProcessing()
    project_dir = "/home/harshil/Harshil/gt/spring2020/research2/burmese-NLP/"
    # src_file = "data/burmese_wiki/wiki_data_1.json"
    # target_file = "data/burmese_wiki/wiki_data_1_processed.json"
    src_file = "data/bible/bible_data.json"
    target_file = "data/bible/bible_data_processed.json"
    dataProcessor.process_annotated_data(os.path.join(
        project_dir+src_file), os.path.join(project_dir+target_file))
