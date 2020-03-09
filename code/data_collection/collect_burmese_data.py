import urllib
import requests
import bs4
import json
import re
import time
import os
from collections import defaultdict



class DataCollector:
    def __init__(self, max_articles=100):
        self.max_articles = max_articles
        self.links_to_parse = set() 
        self.parsed_links = set()
        self.link_data_obj = defaultdict(lambda:{})
        self.project_dir = "/home/harshil/Harshil/gt/spring2020/research2/burmese-NLP"


    
    def get_embedded_links_from_link(self, link):
        try:
            response = requests.get("https://my.wikipedia.org" +link)
        except:
            return None, None
        html = response.text
        # print(html)
        soup = bs4.BeautifulSoup(html, "html.parser")
        embedded_article_links = [] #links on the current page
        article_text = ""
        print("Link: {}".format(link))
        # Find all the direct children of content_div that are paragraphs
        for p_element in soup.find_all("p"):
            if p_element.find("a"):
                embedded_link = p_element.find("a").get('href') 
                #excluding in page links
                if embedded_link and not re.search(r"(#cite|index.php)", embedded_link):
                    embedded_article_links.append(urllib.parse.unquote(embedded_link))


        return link, embedded_article_links

    def get_links_iteratively(self, start_link):
        self.links_to_parse.add(urllib.parse.unquote(start_link))
        while len(self.links_to_parse) > 0 and len(self.parsed_links) < self.max_articles:
            parsed_link, to_parse_links = self.get_embedded_links_from_link(self.links_to_parse.pop())
            # print(to_parse_links)
            if parsed_link and to_parse_links:
                self.links_to_parse.update(to_parse_links)
                self.parsed_links.add(parsed_link)
                time.sleep(2)
        
    def get_data_from_link(self, link):
        try:
            response = requests.get("https://my.wikipedia.org" +link)
        except:
            return None
        
        html = response.text
        soup = bs4.BeautifulSoup(html, "html.parser")
        text = ""
        for p_element in soup.find_all("p"):
            # print(type(p_element.get_text()))
            text += p_element.get_text()
        self.link_data_obj[link]["text-data"] = text 


    def get_data_urls_json(self, url_file, target_file):
        with open(os.path.join(project_dir, url_file), "r+") as url_list_json:
            try:
                links = json.load(url_list_json)
                for i, link in enumerate(links):
                    print("{}-{}".format(i,link))
                    self.get_data_from_link(link)
                    time.sleep(1)
                self.print_data_to_file(target_file)
            except json.decoder.JSONDecodeError:
                print("No urls in the file")

    
    def print_data_to_file(self, file_name):
        with open(os.path.join(self.project_dir, file_name), "r+") as link_data_json:
            try:
                # print(data_collector.parsed_links)
                existing_content = json.load(link_data_json)
                existing_content.update(self.link_data_obj)
                link_data_json.seek(0)
                link_data_json.truncate()
                json.dump(existing_content, link_data_json, indent=4, ensure_ascii=False)
            except json.decoder.JSONDecodeError:
                json.dump(self.link_data_obj, link_data_json, indent=4, ensure_ascii=False)


    def print_links_to_file(self, file_name):
        with open(os.path.join(self.project_dir, file_name), "r+") as url_list_json:
            try:
                # print(data_collector.parsed_links)
                existing_content = json.load(url_list_json)
                union_set = set(existing_content).union(self.parsed_links)
                url_list_json.seek(0)
                url_list_json.truncate()
                json.dump(list(union_set), url_list_json, indent=4, ensure_ascii=False)
            except json.decoder.JSONDecodeError:
                json.dump(list(self.parsed_links), url_list_json, indent=4, ensure_ascii=False)
        





if __name__ == "__main__":
    project_dir = "/home/harshil/Harshil/gt/spring2020/research2/burmese-NLP"
    wiki_url_file = "text/data/burmese_wiki/wiki_urls.json"
    wiki_data_file = "text/data/burmese_wiki/wiki_data.json"
    # link = "/wiki/ဖေ့စ်ဘွတ်ခ်"
    link1 = "/wiki/အက်ဒွင်_လတ္တယင်"
    data_collector = DataCollector()

    
    #collect urls 
    data_collector.get_links_iteratively(link1)

    #print urls to file
    data_collector.print_links_to_file(wiki_url_file)

    #use printed urls to get text data and store it in wiki_data_file
    data_collector.get_data_urls_json(wiki_url_file, wiki_data_file)
    



    

    

        


    