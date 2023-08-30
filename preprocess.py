import  os
path_img=r'/home/visio-ai/PycharmProjects/FaceAndPlate/Dataset/seg/JPEGImages/'
path_mask=r'/home/visio-ai/PycharmProjects/FaceAndPlate/Dataset/seg/SegmentationClass/'

def delete_imgs(path_mask,path_img):
    mask_list = os.listdir(path_mask)
    img_list = os.listdir(path_img)
    for img in img_list:
        mask_name = img.replace(".jpg", ".png")
        if mask_name in mask_list:
            # print(mask_name)
            continue
        else:
            print("deleted " + mask_name)
            os.remove(path_img + img)