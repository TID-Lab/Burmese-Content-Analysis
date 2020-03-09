import json

with open("../data/word_list.json", "r+", encoding = "utf-8") as word_list_by_src_f:
    word_list_by_src = json.load(word_list_by_src_f)
    for j, word_by_src in enumerate(word_list_by_src):
        word_list = word_by_src["words"]
        for i, word in enumerate(word_list):
            word_list[i] = word.rstrip('\n')
        word_by_src["words"] = word_list
    json.dump(word_list_by_src, word_list_by_src_f, indent = 4, ensure_ascii=False)
