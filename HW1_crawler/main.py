from crawler import MyCrawler

if __name__ == "__main__":

    seed_url_list = []
    with open("seed_url.txt") as file_in:
        seed_url_lines = file_in.readlines()
        for line in seed_url_lines:
            seed_url_list.append(line.strip())

    stop_words_list = ['\n', ' ', '\r\n', '\t', '\xa0', '©', '\u200c']
    with open("stop_words.txt", "r", encoding="utf-8") as file_in:
        for line in file_in.readlines():
            stop_words_list.append(line[:-1])

    my_crawler = MyCrawler(20000, seed_url_list, stop_words_list, "result.txt")
    my_crawler.get()
