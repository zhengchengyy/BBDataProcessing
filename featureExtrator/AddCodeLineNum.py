file_name = "ACode.txt"
file_write_obj = open("AResult.txt", 'w')

# 统计一个代码的行数，暂时不用
def countLine(fname):
    count = 0
    for file_line in open(fname, encoding='UTF-8').readlines():
        if file_line != '' and file_line != '\n' and file_line.lstrip()[0]!='#':  # 过滤掉空行、注释
            count += 1
    return count

# 添加编号
line_num = 0
for file_line in open(file_name, encoding='UTF-8').readlines():
    if(line_num < 10):
        code = " " + str(line_num) + ": " + file_line
    else:
        code = str(line_num) + ": " + file_line
    print(code)
    line_num += 1
    # 把格式化代码存入文件
    file_write_obj.write(code)

file_write_obj.close()
