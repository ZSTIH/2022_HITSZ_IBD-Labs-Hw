import re

# 对result.txt进行处理，生成new_result.txt
# 去除非中文字符，并改为10个词一行
# 使用前记得将result.txt的编码改为utf-8

with open("result.txt", "r", encoding="utf-8") as f_in:
    lines = f_in.readlines()

with open("new_result.txt", "a", encoding="utf-8") as f_out:
    i = 0
    for line in lines:
        i += 1
        print(i)
        words = line.split(" ")
        count_to_ten = 0
        for word in words:
            pattern = re.compile(r'[^\u4e00-\u9fa5]')
            word = re.sub(pattern, '', word)
            f_out.write(word)
            count_to_ten += 1
            if count_to_ten == 10:
                count_to_ten = 0
                f_out.write("\n")
            else:
                f_out.write(" ")
