import re
import jieba
import requests
from bs4 import BeautifulSoup
from zhon.hanzi import punctuation as chi_punctuation
from string import punctuation as en_punctuation


class MyCrawler:
    REGEX = "\\b((?:https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:, .;]*[-a-zA-Z0-9+&@#/%=~_|])"
    TIME_OUT_SHORT = 0.1
    TIME_OUT_LONG = 0.5

    def __init__(self, total_number, seed_urls_list, stop_words_list, result_filename):
        self.total_number = total_number
        self.seed_urls_list = seed_urls_list
        self.stop_words_list = stop_words_list
        self.result_filename = result_filename

    def __parse_text(self, string):
        word_list_final = []
        for i in chi_punctuation:
            string = string.replace(i, '')
        for i in en_punctuation:
            string = string.replace(i, '')
        string = re.sub('[a-zA-Z]', '', string)
        string = re.sub('[0-9]', '', string)
        word_list = jieba.lcut(string)
        for word in word_list:
            if word not in self.stop_words_list:
                word_list_final.append(word)
        return " ".join(word_list_final)

    def get(self):
        num = 0
        unvisited_url_list = []
        visited_url_list = []
        while True:

            if len(self.seed_urls_list) != 0:
                element = self.seed_urls_list.pop(0)
                time_out = self.TIME_OUT_LONG
            elif len(unvisited_url_list) != 0:
                element = unvisited_url_list.pop(0)
                time_out = self.TIME_OUT_SHORT
            else:
                print(
                    "The crawler stops working because there are no more accessible web pages! %d web pages have been "
                    "crawled." % num)
                break
            if element in visited_url_list:
                continue
            visited_url_list.append(element)

            try:
                r = requests.get(element, timeout=time_out)
                r.encoding = r.apparent_encoding
                str_html = r.text
                soup = BeautifulSoup(str_html, 'lxml')

                if soup.title is not None:
                    num += 1
                    with open(self.result_filename, "a", encoding=r.encoding) as file_out:
                        file_out.write(self.__parse_text(str_html))

                new_url_list = re.findall(self.REGEX, str_html)
                unvisited_url_list.extend(new_url_list)

                if num == self.total_number:
                    file_out.close()
                    print(
                        "The crawler stops working because the target number has been reached! %d web pages have been "
                        "crawled." % num)
                    break
                elif num > 0 and num % 10 == 0:
                    file_out.close()
                    print("The crawler is now working... %d web pages have been crawled!" % num)

            except Exception as e:
                print("\033[0;31m%s\033[0m" % "Exception!")
                print("\033[0;31m%s\033[0m" % e)
