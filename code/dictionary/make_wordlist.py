import os
import json
import collections
'''
Create a json file from a list of words and generate word statistics.

'''
class WordListMaker():
    def __init__(self):
        pass
    
    def changeEncoding(self, fileName, targetFileName):

        with open(fileName, "r", encoding="utf-8") as sourceFile:
            data = json.load(sourceFile)
            with open(targetFileName, "w") as targetFile:
                json.dump(data, targetFile, indent = 4, ensure_ascii=False)

    def createWordList(self, fileName, newFile):
        if newFile:
            with open(fileName, "r") as f:    
                text = f.readlines()
                src = text[0]
                words = text[1:]
        
        with open("word_list.json", "r+") as word_list_f:
            word_list_by_src = json.load(word_list_f)
            if newFile: word_list_by_src.append({"source": src, "words":words})    
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

    #Takes long time to run.
    def wordStatistics(self):
        word_stats_obj = {}
        reverse_mapping = {}
        word_id = 0
        unique_words  = 0
        unique_words_by_src = {}
        with open("../data/all_words.json", "r", encoding="utf-8") as all_words_f:
            all_words = json.load(all_words_f)
            unique_words = len(all_words)
            with open("../data/word_list.json", "r", encoding="utf-8") as word_list_f:
                word_list_by_src = json.load(word_list_f)
                for word in all_words:
                    word_id += 1
                    reverse_mapping[word_id] = word
                    for word_src_obj in word_list_by_src:
                        src = word_src_obj["source"]
                        words = word_src_obj["words"]
                        unique_words_by_src[src] = len(words)
                        for word_c in words:
                            if word == word_c:
                                if word_id not in word_stats_obj:
                                    word_stats_obj[word_id]={"count":1, "source":[src]}
                                else:
                                    word_stats_obj[word_id]["count"] += 1
                                    word_stats_obj[word_id]["source"].append(src)
                                break
        with open("../analysis/word_stats.txt", "w") as word_stats_f:
            word_stats_f.write("Total unique words : {}\n".format(unique_words))
            word_stats_f.write("Unique words by source:\n")
            for k, v in unique_words_by_src.items():
                word_stats_f.write("\t{} - {}\n".format(k, str(v)))
            word_stats_f.write("\n\n\n")
            
            word_stats_obj = {k: v for k, v in sorted(word_stats_obj.items(), key=lambda item: item[1]["count"])}
            with open("../analysis/word_stats_obj.json", "w", encoding="utf-8") as word_stats_obj_f:
                json.dump(word_stats_obj, word_stats_obj_f, indent=4, ensure_ascii=False)
            word_dict_by_count = collections.defaultdict(list)
            for k, v in list(word_stats_obj.items()):
                word_dict_by_count[v["count"]].append(reverse_mapping[k])  
            with open("../analysis/word_list_by_count.json", "w", encoding="utf-8") as word_list_by_count_f:
                json.dump(word_dict_by_count, word_list_by_count_f, indent=4, ensure_ascii=False)
            for k, v in word_dict_by_count.items():
                word_stats_f.write("Number of words appearing in {} dictionary : {}\n".format(k, len(v)))
            word_stats_f.write("Least 100 used words\n")
            for k, v in list(word_stats_obj.items())[:100]:
                word_stats_f.write("Word   : {} \nCount  : {} \nSource : {}\n".format(reverse_mapping[k], str(v["count"]), str(v["source"])))
                word_stats_f.write("-"*100 + "\n")
            word_stats_f.write("\n\n\n")
            word_stats_f.write("Max 100 used words\n")
            word_stats_obj = {k: v for k, v in sorted(word_stats_obj.items(), key=lambda item: item[1]["count"], reverse=True)}
            for k, v in list(word_stats_obj.items())[:100]:
                word_stats_f.write("Word   : {} \nCount  : {} \nSource : {}\n".format(reverse_mapping[k], str(v["count"]), str(v["source"])))
                word_stats_f.write("-"*100 + "\n")
        
            



        


if __name__ == "__main__":
    wordListMaker = WordListMaker()
    # wordListMaker.createWordList("b7.txt")
    wordListMaker.wordStatistics()
