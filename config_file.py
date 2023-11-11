import argparse


def set_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--github_link', type=str, default="https://github.com/LGNWJQ/", help="Author's github repository")

    # 数据集相关
    parser.add_argument('--training_dataset_path', type=str, default='D:/IEEE_SPL/HDR_DATA/TRAIN/', help='训练集的路径')
    parser.add_argument('--val_dataset_path', type=str, default='D:/IEEE_SPL/HDR_DATA/TEST/', help='验证集的路径')
    parser.add_argument('--patch_size', type=int, default=256, help='训练时图像裁剪patch大小')
    parser.add_argument('--val_patch_size', type=int, default=512, help='验证时图像裁剪patch大小')
    parser.add_argument('--batch_size', type=int, default=4, help='训练批量大小')
    parser.add_argument('--val_batch_size', type=int, default=4, help='验证批量大小')
    parser.add_argument('--edge_type', type=str, default='ST', help='边缘通道类型，可选None, ST, IG')
    parser.add_argument('--num_workers', type=int, default=2, help='读取数据集的线程数量')

    # 训练参数
    parser.add_argument('--exp_name', type=str, default='test_log', help='实验名字')
    parser.add_argument('--basic_channels', type=int, default=64, help='卷积层的特征通道数')
    parser.add_argument('--depths', type=list, default=[2, 2, 2], help='HDR重建网络的每一层的残差快堆叠数量，长度为3的列表')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=2000, help='最大训练周期数')
    parser.add_argument('--min_epochs', type=int, default=1000, help='最小训练周期数')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=10, help='每训练n个周期数检查一次')
    parser.add_argument('--ckpt_path', type=str, default=None, help='预训练权重路径')

    config = parser.parse_args()

    # 显示参数
    # print('=-' * 30)
    # for arg in vars(config):
    #     print('--', arg, ':', getattr(config, arg))
    # print('=-' * 30)

    return config