import os
import json


class WordListMaker():
    def __init__(self):
        pass
    
    def createWordList(self, fileName):

        # with open(fileName, "r") as f:    #only uncomment this block when adding new words from new file.
        #     text = f.readlines()
        #     src = text[0]
        #     words = text[1:]
        
        with open("word_list.json", "r+") as word_list_f:
            word_list_by_src = json.load(word_list_f)
            # word_list_by_src.append({"source": src, "words":words})    #only uncomment when adding new words from new file
            for i, word_src_obj in enumerate(word_list_by_src):
                print("Source: {}".format(word_src_obj["source"]))
                print("Word count before: {}".format(len(word_src_obj["words"])))
                words = set(word_src_obj["words"])
                print("Word count after: {}".format(len(words)))
                word_list_by_src[i]["words"] = list(words)
            with open("all_words.json", "r+") as all_words_f:
                all_words = json.load(all_words_f)
                new_words_arr = list(set(all_words).union(words))
                all_words_f.seek(0)
                json.dump(new_words_arr, all_words_f, indent=4)
                all_words_f.truncate()
            word_list_f.seek(0)        
            json.dump(word_list_by_src, word_list_f, indent=4)
            word_list_f.truncate()     



        
            



        


if __name__ == "__main__":
    wordListMaker = WordListMaker()
    wordListMaker.createWordList("b7.txt")
