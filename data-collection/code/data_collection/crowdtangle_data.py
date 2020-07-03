import requests
import urllib
import time
import json
url = "https://api.crowdtangle.com/posts"

searchTerms = [
"စောက်သုံးမကျတဲ့ကောင်တွေ",
"ငမွဲတွေ",
"စပတ်",
"အဖြူ ကောင်",
"အကန်းတွေ",
"အကန်း",
"စောက်သုံးမကြတဲ့ကောင်များ",
"လူစဉ် မမှီ",
"အဖြူကောင်များ",
"ဆန် ကုန် မြေ လေး",
"ဘောပြား",
"နားလေးတွေ",
"ခွေးမများ",
"အကြိုး",
"နားကန်းတွေ",
"တောသား",
"ငမွဲ",
"ခွေးမျို",
"ဆန်ကုန်မြေလေးတွေ",
"ရှမ်းပုတ်များ",
"မွတ်ကုလားစုတ်",
"ခိုးဝင်ဘင်္ဂလီ",
"စောက် သုံးမကြ",
"တောသားတွေ",
"အဖြူကောင်တွေ",
"အကန်းများ",
"ခွေး ကောင်",
"အရူးတွေ",
"ကုလားကြီး",
"အသားမဲ",
"စောက်ဗမာ",
"ခွေးကုလား",
"ခွေးမတွေ",
"လူလိမ်တွေ",
"ဆန်ကုန်မြေလေးများ",
"လအများ",
"နားလေးများ",
"အကြိုးတွေ",
"အအ",
"လူမျိုးလိမ်",
"မွတ်စုတ်",
"ကုလားစုတ်",
"အအတွေ",
"နားထိုင်",
"နား လေး",
"အသားမဲတွေ",
"ရိုလိမ်ညာ",
"ခွေးကောင်များ",
"အရူးများ",
"နားထိုင်းများ",
"ကုလားဆိုး",
"စောက်ကုလား",
"အအများ",
"စောက်ဖုတ်",
"စပတ်တွေ",
"နားကန်းများ",
"မျိုးမစစ်",
"ရမြေောကမျးတငျ လှစေီးခိုးဝငျ",
"လအ",
"ရှမ်းပုတ်",
"အရူး",
"လအတွေ",
"ဧည့်ဆိုး ကုလား",
"ဘင်္ဂါလီအကြမ်းဖက်",
"အကြိုးများ",
"ရိုဂိန်ညာ",
"ခွေးကောင်တွေ",
"အသားမဲများ",
"ငမွဲများ",
"လူစဉ်မမှီတဲ့ကောင်များ",
"ခွေးမ",
"နားထိုင်းတွေ",
]



searchTerms2 = [
    "ဘင်္ဂလီ မွတ်စလင်","ရေမြောကမ်းတင်","ဘုရင် ဂျီ","ဘုရင်ဂျီတွေ","မျိုးချစ်တွေ","အစွန်းရောက်တွေ","တောသားများ","ရခီးများ","အ ခြောက်","မျိုးချစ်များ","ဘင်္ဂလီ","တရားသောစစ်ပွဲ","ကုလားတွေ","မောဂ်ကုလား","လူလိမ်များ","အ ရှေ့တိုင်း","ဘင်္ဂလီများ","အစွန်း ရောက်","ဘင်္ဂါလီ အကြမ်းဖက်သမား","မိန်းမလျာများ","ရှမ်းပုတ်တွေ","ဒုက္ခိတ","ဘုရင်ဂျီများ","အစွန်းရောက်များ","မျိုးချစ်","မိန်းမလျာ","ကုလားဖြူများ","အနောက်တံခါး","ကပြားများ","mုလား","ဘင်္ဂါလီ","မူဂျာဟစ်","လူလိမ်","မွတ်ကုလား","မိန်းမလျများ","အခြောက်တွေ","မိန်းမလျာတွေ","လူစဉ်မမှီတဲ့ကောင်တွေ","အခြောက်မျာ","မွတ်စလင်","ကုလားများ","ကုလားဖွူ","ကုလားဖြူ","ကုလားဖြူတွေ","ဘင်္ဂလီတွေ","ရခီး","ဒုက္ခိတများ","ကပြားတွေ","ဒုက္ခိတတွေ","ကုလား","ကပြား","နားကန်း","ရခီးတွေ"
]
lexicon_dict = {}
for i, term in enumerate(searchTerms2):
    lexicon_dict[term] = {
        "id": i,
        "posts":[]
    }

paginated_responses = []

responses = []


for i, term in enumerate(searchTerms2):
    if (i+1)%5 == 0:
        print("sleeping")
        time.sleep(60)
    html_encoded_string = urllib.parse.quote(term)
    querystring = {"searchTerm":html_encoded_string,"startDate":"2019-05-13"}

    headers = {
        'x-api-token': "T20RoAVir0jWCmJRYLax2LsGhUfuIptSqyYW6u7l"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)
    response_json = json.loads(urllib.parse.unquote(response.text))
    if "nextPage" in response_json["result"]["pagination"]:
        paginated_responses.append(response_json["result"]["pagination"]["nextPage"])
    

    print(response_json)
    responses.append(response_json)
    lexicon_dict[term]["posts"].append(response_json)

print(paginated_responses)
with open("ct-data.2.json", "w", encoding="utf-8") as ct_data_file:
    json.dump(responses, ct_data_file, indent=4, ensure_ascii=False)
with open("ct-links.2.json", "w", encoding="utf-8") as ct_link_file:
    json.dump(paginated_responses, ct_link_file, indent=4, ensure_ascii=False)

with open("ct-lex-dict.2.json", "w", encoding="utf-8") as ct_lex_file:
    json.dump(lexicon_dict, ct_lex_file, indent=4, ensure_ascii=False)
