import os
import json


class WordListMaker():
    def __init__(self):
        pass
    
    def createWordList(self):

        # formatting raw words
        #
        # with open(fileName, "r") as f:
        #     text = f.readlines()
        #     src = text[0]
        #     words = text[1:]
        # for i,word in enumerate(words):
        #     words[i] = word.split(",")[0]
        # with open(fileName, "w") as f:
        #     f.write(src)
        #     for word in words:
        #         f.write(word+"\n")
        
        with open("word_list.json", "r+") as word_list_f:
            word_list_by_src = json.load(word_list_f)
            for i, word_src_obj in enumerate(word_list_by_src):
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
            word_list_f.seek(0)        # <--- should reset file position to the beginning.
            json.dump(word_list_by_src, word_list_f, indent=4)
            word_list_f.truncate()     






        # word_list.append({"source": src, "words":words})
        # json.dump(word_list, open("word-list.json", "w"))

        
            



        


if __name__ == "__main__":
    wordListMaker = WordListMaker()
    wordListMaker.createWordList()
