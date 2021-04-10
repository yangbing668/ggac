# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:14:45 2020

@author: Barry
"""

#提取目录下所有图片,更改尺寸后保存到另一目录
#from PIL import Image
#import os.path
#import glob
#def convertjpg(jpgfile,outdir,width=256,height=256):
#    img=Image.open(jpgfile)
#    try:
#        new_img=img.resize((width,height),Image.BILINEAR)
#        new_img=new_img.crop((16,16,240,240))
#        new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
#    except Exception as e:
#        print(e)
#
#num = 0
#for jpgfile in glob.glob(('H:/images/images/*.png')):
#    convertjpg(jpgfile,"F:/医疗数据/ChestX-ray14/images/cropedimages")
#    num += 1
#    print('已处理个数：',num,'当前图片名：',jpgfile)

#作者提供的方法
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

transform=transforms.Compose([transforms.Resize((256,256)), transforms.CenterCrop((224,224))])
df = pd.read_csv('F:/医疗数据/ChestX-ray14/Data_Entry_2017.csv')

num = 73000
for name in df['Image Index'][num:]:
    path = os.path.join('H:/images/images', name)
    image = Image.open(path)
    transform(image).save('F:/医疗数据/ChestX-ray14/images/cropedimages/'+name)
    num += 1
    print(num,name)






