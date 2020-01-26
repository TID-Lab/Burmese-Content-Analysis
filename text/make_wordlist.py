import os
import json


class WordListMaker():
    def __init__(self):
        pass
    
    def createWordList(self, fileName):

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
        
        # word_list = json.load(open("word-list.json", "r"))
        # word_list.append({"source": src, "words":words})
        # json.dump(word_list, open("word-list.json", "w"))

        
            



        


if __name__ == "__main__":
    wordListMaker = WordListMaker()
    # wordListMaker.createWordList("b1.txt")
