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
    
    
    # 1 Loading weights
    for i in range(len(weight_wait_list)):
        test_model._load_weights(weight_wait_list[i])
        saved_str = weight_wait_list[i].split("\\")[1].split("_")[0]
        test_model.predict("asset/original_pictures/chinese_text.png","asset/chinese_text"+saved_str+".png")
        test_model.predict("asset/original_pictures/ut_austin_econ.jpg","asset/ut_austin_econ"+saved_str+".jpg")    
        test_model.predict("asset/original_pictures/github.jpg","asset/github"+saved_str+".jpg")    


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
    