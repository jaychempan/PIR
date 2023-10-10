import json
import os
import cv2
# import requests
# import numpy as np
# import matplotlib.pyplot as plt
def read_dataset(img_path, json_path, filename):
    dataset = json.load(open(json_path))
    # print(dataset['images'])
    sentences = []
    imgid = -1
    for i in dataset['images']:
        if i['filename'] == filename:
            imgid = i['imgid']
            for ii in i['sentences']:
                sentences.append(ii['raw'])
            break
    print('image_id: {}'.format(imgid))
    print('file_name: {}'.format(filename))
    d = 0
    for aa in sentences:
        d += 1
        print('caption_{}: {}'.format(d, aa))
    pt = os.path.join(img_path, filename)
    print(pt)
    img = cv2.imread(pt)
    cv2.imshow(filename, img)
    cv2.waitKey(0)  # 按键结束
    cv2.destroyAllWindows()
def get_img_filename(path, rank):
    # print(data_path)
    with open(path, encoding='utf-8') as file:
        content = file.readlines()

    return str(content[rank].rstrip())


if __name__ == "__main__":
    ch = input('input the num of dataset => 1. rsicd  2. rsitmd  3. Sydney  4. UCM: \n')
    if ch == '1':
        rank = int(input("Please enter the number of images you want to query in the data set: \n"))
        filename = get_img_filename(path = '/data/pjc/CMR-Pro/GaLR/data/rsicd_precomp/train_filename.txt', rank=rank - 1)
        read_dataset(img_path = '/data/pjc/CMR-Pro/rs_data/rsicd/images',json_path = '/data/pjc/CMR-Pro/rs_data/rsicd/dataset_rsicd.json', filename = filename)
    elif ch == '2':
        rank = int(input("Please enter the number of images you want to query in the data set: \n"))
        filename = get_img_filename(path = '/data/pjc/CMR-Pro/GaLR/data/rsitmd_precomp/train_filename.txt', rank=rank - 1)
        read_dataset(img_path = '/data/pjc/rs_data/rsitmd/images',json_path = '/data/pjc/rs_data/rsitmd/dataset_RSITMD.json', filename = filename)
    else:
        print("wrong input !")
