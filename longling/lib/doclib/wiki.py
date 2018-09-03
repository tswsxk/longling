# coding: utf-8
# create by tongshiwei on 2018/9/3

def _beautify(row):
    return [' ' + d + ' ' for d in row]

def csv_string_to_table(long_string, delimiter="\t"):
    output = ""
    print('{| class="wikitable"')
    row_data = [list(map(lambda x: x.strip(), _.split(delimiter))) for _ in long_string.strip().split("\n") if _]
    print("|-\n!" + "!!".join(_beautify(row_data[0])))
    for row in row_data[1:]:
        print("|-\n|" + "||".join(_beautify(row)))
    print("|}")

if __name__ == '__main__':
    long_string = '''
705座位表			
706门			705门
	沈双宏	刘杨	
		王仲	
	赵伟豪	李徵	程亦飞
			
			

    '''
    csv_string_to_table(long_string)
