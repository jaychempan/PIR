from read_dataset import *
from loc_bbox import *

rank = int(input("Please enter the number of images you want to query in the data set: \n"))
img_id = get_img_id(root='/data/pjc/CMR-Pro/data', dataset_name='coco', split='train', rank=rank - 1)
read_dataset(root='/data/pjc/data', dataset_name='coco', split='train', img_id=img_id)
loc_bbox(coordinate_path='/data/pjc/CMR-Pro/ITR/bbox_data/coco_precomp/train_ims_bbx.npy', rank=rank - 1)