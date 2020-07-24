import tensorflow as tf
from glob import glob
from ctpn_tools.core import CTPN
from ctpn_tools.data_loader import DataGenerator
import warnings
warnings.filterwarnings("ignore")


def parser(my_dic):
    result = []
    for key in my_dic.keys():
        result.append(key+"_"+str(my_dic[key]))
    
    return "_".join(result)
    
    
if __name__ == '__main__':

    # 0 Prepare for config
    train_model = CTPN()


    # 1 Prepare for data
    anno_dir = r"data\Annotations"
    images_dir = r"data\JPEGImages"
    main_data = glob(anno_dir+ '/*.xml')

        
    # 2 Start training
    for i in range(50):
        dll = DataGenerator(main_data)
        it_dll = iter(dll)
        lr = 0.0001 if i<=30 else 0.00001
        train_model._compile_net(lr)
        his = train_model.core.fit_generator(it_dll,steps_per_epoch=dll.__len__(),epochs=1,callbacks = [tf.keras.callbacks.History()])
        saved_str = "weights/"+str(i)+"_"+parser(his.history)+".h5"
        train_model.train_model.save_weights(saved_str)
    
    