from ctpn_tools.core import CTPN  
from glob import glob
import os
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def renew(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.mkdir(path)


if __name__=="__main__":
    
    # 0 Define pictures,weights and test_model
    weight_wait_list = glob("weights/*loss*loss*")
    test_model = CTPN()
    pictures = glob('asset/original_pictures/*')
    
    
    # 1 Load weights
    for weight in weight_wait_list:
        weight_num = int(weight.split("\\")[1].split("_")[0])
        if weight_num<=19:
            continue
        test_model._load_weights(weight)
        my_path = "asset/weight_"+str(weight_num)
        renew(my_path)
        for pic in pictures:
            pic_name = pic.split("\\")[1]
            test_model.final_predict(pic, my_path+"/"+pic_name)









    #(1, 600, 800, 3) #img
    #(1, 1, 18500) #rpn_class
    #(1, 18500, 3) #rpn_regress
    