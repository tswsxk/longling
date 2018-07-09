# coding: utf-8

'''
@ tesseract
brew install tesseract

@ guide
https://pypi.python.org/pypi/pytesseract
http://www.cnblogs.com/wzben/p/5930538.html
http://www.cnblogs.com/wzben/p/5930538.html
http://www.pyimagesearch.com/2017/07/10/using-tesseract-ocr-python/
'''
import os
import subprocess
import logging

logger = logging.getLogger('ocr')
formatter = logging.Formatter('%(asctime)s, %(levelname)s %(message)s')
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

def image_to_string(img, cleanup=True, plus=''):
    # cleanup为True则识别完成后删除生成的文本文件
    # plus参数为给tesseract的附加高级参数
    if not os.path.exists(img):
        logger.error("%s doesn't exist", img)
        return ''
    subprocess.check_output('tesseract ' + img + ' ' +
                            img + ' ' + plus, shell=True)  # 生成同名txt文件
    text = ''
    with open(img + '.txt', 'r') as f:
        text = f.read().strip()
    if cleanup:
        os.remove(img + '.txt')
    return text


if __name__ == '__main__':
    # try:
    #     import Image
    # except ImportError:
    #     from PIL import Image
    # import pytesseract
    #
    # pytesseract.pytesseract.tesseract_cmd = '/usr/local/Cellar/tesseract/3.05.01/'
    # # Include the above line, if you don't have tesseract executable in your PATH
    # # Example tesseract_cmd: 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract'
    #
    # print(pytesseract.image_to_string(Image.open('test.png')))
    # print(pytesseract.image_to_string(Image.open('test.png'), lang='fra'))

    # print(image_to_string('./test.png', True, '-l chi_sim'))
    print(image_to_string('./test5.png'))



