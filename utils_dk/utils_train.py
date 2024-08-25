import os
import torch
import datetime
import numpy as np
import torch.nn as nn

import sklearn.metrics as metrics
from .utils_modules import init_weight, HintLoss, dkd_loss


def train(model, loader_train, loader_valid, epochs, optimizer):
    # 超参数设置
    beta = 5.0
    alpha = 1.0
    temperature = 2

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    crit_hidden_loss = HintLoss()

    # 建立文件夹保存权重
    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    # 保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 参数初始化
    train_loss = 0
    best_model = 0

    loss_list_train = []
    loss_list_valid = []

    accuracy_list_train = []
    accuracy_list_valid = []

    # cos 学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, T_mult=2)

    for epoch in range(epochs):

        # 初始化准确率
        train_avg_loss = 0
        valid_avg_loss = 0
        train_accuracy = 0
        valid_accuracy = 0

        # 预测标签和真实标签存储
        train_score_list = []
        train_label_list = []
        valid_score_list = []
        valid_label_list = []

        '''模型训练 '''
        model.train()
        for i, (train_images, train_labels) in enumerate(loader_train):
            # 添加到 CUDA 中
            train_images, train_labels = train_images.cuda(), train_labels.cuda()

            # 梯度清零
            optimizer.zero_grad()

            # 获取输出
            train_out = model(train_images)

            # 预测输出
            train_predicts_1 = train_out["Y1"]
            train_predicts_2 = train_out["Y2"]

            # 注意力输出
            t_focal_attention_1 = train_out["FA_1"]
            t_focal_attention_2 = train_out["FA_2"]

            t_global_attention_1 = train_out["GA_1"]
            t_global_attention_2 = train_out["GA_2"]
            t_global_attention_3 = train_out["GA_3"]
            t_global_attention_4 = train_out["GA_4"]

            # 监督损失
            train_loss_1 = criterion(train_predicts_1, train_labels.long(), )
            train_loss_2 = criterion(train_predicts_1, train_labels.long())

            # 焦点损失
            t_focal_loss = crit_hidden_loss(t_focal_attention_1, t_focal_attention_2)

            # 全局损失
            t_global_loss_1 = crit_hidden_loss(t_global_attention_1, t_global_attention_2)
            t_global_loss_2 = crit_hidden_loss(t_global_attention_3, t_global_attention_4)
            t_global_loss = t_global_loss_1 + t_global_loss_2

            # 蒸馏损失
            label_loss = dkd_loss(train_predicts_1, train_predicts_2, train_labels,
                                  alpha=alpha, beta=beta, temperature=temperature)

            # 损失合并
            train_loss = label_loss + train_loss_1 + train_loss_2 + t_focal_loss + t_global_loss

            # 反向传播
            train_loss.backward()
            optimizer.step()

            # 计算准确率和平均损失
            train_predict = (train_predicts_1 + train_predicts_2).detach().max(1)[1]
            train_mid_acc = torch.as_tensor(train_labels == train_predict)
            train_accuracy = train_accuracy + torch.sum(train_mid_acc) / len(train_labels)

            train_avg_loss = train_avg_loss + train_loss / len(loader_train)

            # 存储预测值和真实值
            train_score_list.extend(train_predict.cpu().numpy())
            train_label_list.extend(train_labels.cpu().numpy())

        # 更新学习率
        scheduler.step()

        '''模型测试 '''
        with torch.no_grad():

            model.eval()
            for i, (valid_images, valid_labels) in enumerate(loader_valid):
                # 添加到 CUDA 中
                valid_images, valid_labels = valid_images.cuda(), valid_labels.cuda()

                # 获取标签
                valid_out = model(valid_images)

                # 预测输出
                valid_predicts_1 = valid_out["Y1"]
                valid_predicts_2 = valid_out["Y2"]

                # 注意力输出
                v_focal_attention_1 = valid_out["FA_1"]
                v_focal_attention_2 = valid_out["FA_2"]

                v_global_attention_1 = valid_out["GA_1"]
                v_global_attention_2 = valid_out["GA_2"]
                v_global_attention_3 = valid_out["GA_3"]
                v_global_attention_4 = valid_out["GA_4"]

                # 混合监督损失
                valid_loss_1 = criterion(valid_predicts_1, valid_labels.long())
                valid_loss_2 = criterion(valid_predicts_2, valid_labels.long())

                # 焦点损失
                v_focal_loss = crit_hidden_loss(v_focal_attention_1, v_focal_attention_2)

                # 全局损失
                v_global_loss_1 = crit_hidden_loss(v_global_attention_1, v_global_attention_2)
                v_global_loss_2 = crit_hidden_loss(v_global_attention_3, v_global_attention_4)
                v_global_loss = v_global_loss_1 + v_global_loss_2

                # 蒸馏损失
                label_loss = dkd_loss(valid_predicts_1, valid_predicts_2, valid_labels,
                                      alpha=alpha, beta=beta, temperature=temperature)

                # 损失合并
                valid_loss = label_loss + valid_loss_1 + valid_loss_2 + v_focal_loss + v_global_loss

                # 计算准确率和平均损失
                valid_predict = (valid_predicts_1 + valid_predicts_2).detach().max(1)[1]
                valid_mid_acc = torch.as_tensor(valid_labels == valid_predict)
                valid_accuracy = valid_accuracy + torch.sum(valid_mid_acc) / len(valid_labels)

                valid_avg_loss = valid_avg_loss + valid_loss / len(loader_valid)

                # 存储预测值和真实值
                valid_score_list.extend(valid_predict.cpu().numpy())
                valid_label_list.extend(valid_labels.cpu().numpy())

            # 记录损失
            loss_list_train.append(train_loss.detach().cpu().item())
            loss_list_valid.append(valid_loss.detach().cpu().item())

            # 记录准确率
            accuracy_list_train.append(train_accuracy.detach().cpu().item() / len(loader_train))
            accuracy_list_valid.append(valid_accuracy.detach().cpu().item() / len(loader_valid))

        # 计算召回率
        train_recall = metrics.recall_score(train_score_list, train_label_list,
                                            labels=np.unique(train_label_list))
        valid_recall = metrics.recall_score(valid_score_list, valid_label_list,
                                            labels=np.unique(valid_label_list))

        # 计算 F1 值
        train_f1_score = metrics.f1_score(train_score_list, train_label_list,
                                          labels=np.unique(train_label_list))
        valid_f1_score = metrics.f1_score(valid_score_list, valid_label_list,
                                          labels=np.unique(valid_label_list))

        # 计算精准率
        train_precision = metrics.precision_score(train_score_list, train_label_list,
                                                  labels=np.unique(train_label_list))
        valid_precision = metrics.precision_score(valid_score_list, valid_label_list,
                                                  labels=np.unique(valid_label_list))

        # 输出结果
        train_avg_loss = train_avg_loss.detach().cpu().item()
        train_accuracy_avg = train_accuracy.detach().cpu().item() / len(loader_train)
        print('训练: Epoch %d, Accuracy %f, Train Loss: %f' % (epoch, train_accuracy_avg, train_avg_loss))

        valid_avg_loss = valid_avg_loss.detach().cpu().item()
        valid_accuracy_avg = valid_accuracy.detach().cpu().item() / len(loader_valid)
        print('验证: Epoch %d, Accuracy %f, Valid Loss: %f' % (epoch, valid_accuracy_avg, valid_avg_loss))

        # 保存最佳验证准确率模型
        if valid_accuracy_avg >= best_model:
            torch.save(model.state_dict(), "save_weights/best_model.pth")
            best_model = valid_accuracy_avg
            print("当前最佳模型已获取")

        # 记录每个 epoch 对应的 train_loss、lr 以及验证集各指标
        with open(results_file, "a") as f:
            info = f"[epoch: {epoch}]\n" \
                   f"train_loss: {train_avg_loss:.6f}\n" \
                   f"valid_loss: {valid_avg_loss:.6f}\n" \
                   f"train_recall: {train_recall:.4f}\n" \
                   f"valid_recall: {valid_recall:.4f}\n" \
                   f"train_F1_score: {train_f1_score:.4f}\n" \
                   f"valid_F1_score: {valid_f1_score:.4f}\n" \
                   f"train_precision: {train_precision:.4f}\n" \
                   f"valid_precision: {valid_precision:.4f}\n" \
                   f"train_accuracy: {train_accuracy_avg:.6f}\n" \
                   f"valid_accuracy: {valid_accuracy_avg:.6f}\n"
            f.write(info + "\n\n")

    # 返回模型
    loss = {'Loss1': loss_list_train, 'Loss2': loss_list_valid,
            'Accuracy1': accuracy_list_train, 'Accuracy2': accuracy_list_valid}

    return model, loss
