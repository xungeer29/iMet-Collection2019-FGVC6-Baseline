# -*- coding:utf-8 -*- 

class DefaultConfigs(object):
    data_root = '/media/gfx/data1/DATA/Kaggle/iMet' # 数据集的根目录
    model = 'ResNet18' # ResNet152 使用的模型
    freeze = True # 是否冻结卷基层

    seed = 1000 # 固定随机种子
    num_workers = 12 # DataLoader 中的多线程数量
    num_classes = 1103 # 分类类别数
    init_epoch = 7 # start epoch
    num_epochs = 200
    batch_size = 128 # 128
    lr = 0.01 # 初始lr
    width = 128 # 输入图像的宽
    height = 128 # 输入图像的高
    iter_smooth = 50 # 打印&记录log的频率

    # OHEM
    OHEM = False
    OHEM_ratio = 0.1

    # resume checkpoint
    resume = True #
    checkpoint = 'ResNet18_python3.pth' # 训练完成的模型名

    # smooth label
    smooth_label = False

config = DefaultConfigs()
