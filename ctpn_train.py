import tensorflow as tf
from glob import glob
from ctpn_tools.core import CTPN
from ctpn_tools.data_loader import DataLoader
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
        dll = DataLoader(main_data)      
        lr = 0.0001 if i<=30 else 0.00001
        train_model._compile_net(lr)
        his = train_model.train_model.fit_generator(dll.load_data(),epochs=1,steps_per_epoch=6000,callbacks = [tf.keras.callbacks.History()])
        saved_str = "weights/"+str(i)+"_"+parser(his.history)+".h5"
        train_model.train_model.save_weights(saved_str)
    
    
    
    
"""    
train_model.train_model.save_weights("weights/0.1044_0.1138.h5")


train_model.train_model.compile(optimizer=SGD(0.0001),
                       loss={'rpn_regress_reshape': _rpn_loss_regr, 'rpn_class_reshape': _rpn_loss_cls},
                       loss_weights={'rpn_regress_reshape': 1.0, 'rpn_class_reshape': 1.0})

"""