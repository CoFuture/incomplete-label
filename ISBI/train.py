from model.unet import UNet
from utils.dataset import CustomDataset, ValidDataset
from torch.utils.data import Dataset
from torch import optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt


def train_net(network, device, train_path, valid_path, epochs=10, batch_size=1, lr=0.005):
    # 加载训练集
    train_dataset = CustomDataset(train_path)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    # 加载验证集合
    valid_dataset = ValidDataset(valid_path)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    # 定义优化器 RMS prop算法
    optimizer = optim.RMSprop(network.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)

    # 定义Loss
    criterion = nn.BCEWithLogitsLoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')

    epoch_list = []
    t_loss_list = []
    v_loss_list = []
    # 训练epochs次
    for epoch in range(epochs):
        train_loss_total = 0.0
        valid_loss_total = 0.0
        train_batch_count = 0
        valid_batch_count = 0
        # 训练模式
        network.train()
        # 按照batch_size开始训练
        for image, label in train_loader:
            # 重置优化器梯度
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = network(image)
            # 计算loss
            loss = criterion(pred, label)
            print('Loss/train', loss.item())
            # 记录总loss
            train_loss_total += loss.item()
            train_batch_count += batch_size

            # 更新参数
            loss.backward()
            optimizer.step()

        # 计算valid loss
        for v_image, v_label in valid_loader:
            # 将数据拷贝到device中
            v_image = v_image.to(device=device, dtype=torch.float32)
            v_label = v_label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            v_pred = network(v_image)
            # 计算loss
            v_loss = criterion(v_pred, v_label)
            print('Loss/valid', v_loss.item())

            valid_loss_total += v_loss.item()
            valid_batch_count += batch_size

            # 保存loss值最小的网络参数
            if v_loss < best_loss:
                best_loss = v_loss
                torch.save(network.state_dict(), 'train_model_epoch40.pth')
                # torch.save(network.state_dict(), 'train_model_epoch60.pth')
                # torch.save(network.state_dict(), 'train_model_in10_epoch40.pth')
                # torch.save(network.state_dict(), 'train_model_in20_epoch40.pth')
                # torch.save(network.state_dict(), 'train_model_in30_epoch40.pth')
                # torch.save(network.state_dict(), 'train_model_in40_epoch40.pth')
                # torch.save(network.state_dict(), 'train_model_in50_epoch40.pth')

        t_loss_list.append(train_loss_total / train_batch_count)
        v_loss_list.append(valid_loss_total / valid_batch_count)

        epoch_list.append(epoch)

    print("Best loss:", best_loss)

    # 绘制图表
    plt.subplot(1, 2, 1)
    plt.plot(epoch_list, t_loss_list)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("train loss-epoch")
    # plt.title("label_in50 training loss-epoch")

    plt.subplot(1, 2, 2)
    plt.plot(epoch_list, v_loss_list)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("valid loss-epoch")

    plt.tight_layout()
    plt.savefig("./log/train_valid_loss_epoch40.png")
    # plt.savefig("./log/train_valid_loss_epoch60.png")
    # plt.savefig("./log/train_valid_loss_epoch20.png")
    # plt.savefig("./log/train_valid_loss_epoch20.png")
    # plt.savefig("./log/train_valid_loss_epoch20.png")
    # plt.savefig("./log/train_valid_loss_epoch20.png")
    # plt.savefig("./log/train_valid_loss_epoch20.png")
    plt.show()


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    r_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道1，分类为1。
    u_net = UNet(in_channels=1, out_classes=1)
    # 将网络拷贝到device中
    u_net.to(device=r_device)
    # 指定训练集地址，开始训练
    t_path = "data/train/"
    v_path = "data/valid/"
    train_net(u_net, r_device, t_path, v_path, epochs=40)
