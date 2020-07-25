#from ctpn_tools.infer_core import CTPN  
from glob import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__=="__main__":
    
    # 0 Define pictures,weights and test_model
    from ctpn_tools.infer_core import CTPN 
    test_model = CTPN()
    test_model._load_weights("weights/CTPN.h5")
    pictures = glob('asset/original_pictures/*')
    
    
    # 1 Load weights
    for pic in pictures:
        my_path = "asset"
        pic_name = pic.split("\\")[1]
        cc = test_model.predict(pic, my_path+"/"+pic_name)



        