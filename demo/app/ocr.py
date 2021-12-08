import os
import sys
import cv2
import easyocr
import argparse
import keras_ocr
# import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
import fastapi
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

image = sys.argv[1]

config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
config['cnn']['pretrained']=False
config['device'] = 'cpu'
config['predictor']['beamsearch']=False
detector = Predictor(config)

def cleanup_text(text):
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()

text_reader = easyocr.Reader(['vi']) 

# url = [
#     "data/cmnd_hng_11_001.jpg",
#     "data/cmnd_hoan_1.jpg",
#     "data/cmnd_Hong_M_Ngan.jpg",
#     "data/cmnd_H_Th_Thu_Tho_.jpg"
# ]
# images = [ keras_ocr.tools.read(i) for i in url]

img = keras_ocr.tools.read(image)
results = text_reader.readtext(img)
for (bbox, text, prob) in results:
    (tl, tr, br, bl) = bbox
    tl = (int(tl[0]), int(tl[1]))
    tr = (int(tr[0]), int(tr[1]))
    br = (int(br[0]), int(br[1]))
    bl = (int(bl[0]), int(bl[1]))
    
    min_x = min(tl[0], tr[0], br[0], bl[0])
    max_x = max(tl[0], tr[0], br[0], bl[0])
    min_y = min(tl[1], tr[1], br[1], bl[1])
    max_y = max(tl[1], tr[1], br[1], bl[1])

    s = detector.predict(Image.fromarray(img[min_y:max_y,min_x:max_x])) 
    print(s)
    # cv2.rectangle(nct, tl, br, (0, 255, 0), 2)
    
Image.fromarray(img[min_y:max_y,min_x:max_x]).show()
# plt.imshow(img)
# cv2.imshow("Image", img)
# cv2.waitKey(0)