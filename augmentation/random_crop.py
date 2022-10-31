import os
from PIL import Image
import random

# file_root = "../images/cracks/Training_Images/"
# file_list = os.listdir(file_root)
# print(file_list)
#
# w=256
# h=256
# #os.mkdir("C:/Users/tjzhang/Documents/TJzhang/Microstructure-GAN-main/crop_image/")
# save_out = "./augment_images/random_crop/"
# for img_name in file_list:
#     img_path = file_root+'/'+img_name
#     im=Image.open(img_path)
#     #im = im.crop((0,0,1221,775))#for the navi image
#     m,n=im.size
#     for i in range(10):
#         x = random.randint(0,m-w)
#         y = random.randint(0,n-h)
#         region = im.crop((x,y,x+w,y+h))
#         region.save(save_out+str(img_name[:-4])+'-'+str(i)+".jpg")


file_root = "../images/cracks/Training_Labels/"
file_list = os.listdir(file_root)
print(file_list)

#os.mkdir("C:/Users/tjzhang/Documents/TJzhang/Microstructure-GAN-main/crop_image/")
save_out = "./augment_images/random_crop/"
i=0
for img_name in file_list:

    img_path = file_root+'/'+img_name
    im=Image.open(img_path)
    #im = im.crop((0,0,1221,775))#for the navi image

    im.save(save_out+str(i).zfill(5)+".png")
    i=i+1