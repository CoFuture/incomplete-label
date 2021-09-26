import glob
import numpy as np
import torch
import os
import cv2
from model.unet import UNet

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = UNet(in_channels=1, out_classes=1)

    # 将网络拷贝到device中
    net.to(device=device)
    # 加载模型参数
    # net.load_state_dict(torch.load('train_model_in25_epoch40.pth', map_location=device))
    # net.load_state_dict(torch.load('train_model_in25_epoch80.pth', map_location=device))
    net.load_state_dict(torch.load('train_model_in50_epoch40.pth', map_location=device))

    # 测试模式
    net.eval()
    # 读取所有图片路径
    tests_path = glob.glob('data/test/*.png')

    # 损失函数
    loss_function = torch.nn.BCEWithLogitsLoss()

    # 遍历素有图片
    for test_path in tests_path:
        # 保存结果地址
        save_res_path = test_path.split('.')[0] + '_res.png'
        # save_res_path = test_path.split('.')[0] + '_i_res.png'
        # save_res_path = save_res_path.replace('test', 'test_in25_epoch40')
        # save_res_path = save_res_path.replace('test', 'test_in25_epoch80')
        save_res_path = save_res_path.replace('test', 'test_in50_epoch40')
        # save_res_path = save_res_path.replace('test', 'test_i_epoch40')
        # 读取图片
        img = cv2.imread(test_path)
        # 转为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 转为batch为1，通道为1，大小为512*512的数组
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
        predict_result = net(img_tensor)

        # todo 计算与complete label的loss
        test_loss = loss_function(predict_result, predict_result)

        # 提取结果
        predict_result = np.array(predict_result.data.cpu()[0])[0]

        # 设置threshold
        threshold = 0.5
        # 处理结果
        predict_result[predict_result >= threshold] = 255
        predict_result[predict_result < threshold] = 0

        # 保存图片
        cv2.imwrite(save_res_path, predict_result)
