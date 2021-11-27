#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib

import BZT_detect
import backdoor

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
import poison
import sim,agg
import datetime
import time
# import backdoor_data

mal_clients = []
for i in range(2):
    mal_clients.append(i)
if __name__ == '__main__':
    f = open("out.txt", "w")  # 打开文件以便写入
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        # dataset_train = backdoor_data.backdoor_data_making()
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # 制作后门数据集
        # dataset_backdoor = backdoor.Export_backdoor_MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_backdoor = backdoor.InfectedMNIST('../data/mnist/', train=False, download=True, transform=trans_mnist, p = 1)
        dataset_backdoor_train = backdoor.InfectedMNIST('../data/mnist/', train=True, download=True, transform=trans_mnist, p = 0.15)
        # sample users
        if args.iid:
            print("iid")
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            print("non iid")
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
        # saved_dic = torch.load('model/mnist_backdoor.pt')
        # net_glob.load_state_dict(saved_dic)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    print(net_glob,file=f)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()
    dic = {}


    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    # 画图用的Acc   list
    Accuracy_list = []


    if args.all_clients: 
        print("Aggregation over all clients")
        print("Aggregation over all clients",file = f)
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        # 程序计时器，启动计时器
        start = time.perf_counter()
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)

        # idxs_users = np.random.choice(range(args.num_users), 5, replace=False)
        # print(idxs_users)
        idxs_users = [0, 1, 2, 3, 4]
        idxs_users = np.array(idxs_users)
        if iter < 3:
            for idx in idxs_users:
                # 全局训练
                local = LocalUpdate(args=args, dataset=dataset_backdoor_train, idxs=dict_users[idx])
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                if args.all_clients:
                    w_locals[idx] = copy.deepcopy(w)
                else:
                    w_locals.app


                    end(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
            # 取出所有的w权重转换为torch模型的形式，以便于使用攻击代码
            for idx in idxs_users:
                net_glob.load_state_dict(w_locals[idx])
                model_temp = copy.deepcopy(net_glob)
                dic[idx] = model_temp

            # do the BZT attack
            # if iter < 3:
            dic = poison.sota_agnostic_min_max(dic, len(mal_clients), f)
        else:
            for idx in idxs_users:
                # 全局训练
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                if args.all_clients:
                    w_locals[idx] = copy.deepcopy(w)
                else:
                    w_locals.app

                    end(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
            # 取出所有的w权重转换为torch模型的形式，以便于使用攻击代码
            for idx in idxs_users:
                net_glob.load_state_dict(w_locals[idx])
                model_temp = copy.deepcopy(net_glob)
                dic[idx] = model_temp

        x_locals = []

        # 将攻击后的模型权重存为w_local的形式，以便于FedAvg
        for model in dic:
            # print(dic[model].state_dict())
            x_locals.append(dic[model].state_dict())
        # 取出所有参与方的模型进行后门数据集的判别
        for idx in idxs_users:
            net_glob.load_state_dict(x_locals[idx])
            model_temp = copy.deepcopy(net_glob)
            model_temp.eval()
            acc_train11, loss_train1 = test_img(model_temp, dataset_backdoor, args)
            acc_test11, loss_test1 = test_img(model_temp, dataset_test, args)
            print("USER "+str(idx)+" BACKDOOR data accuracy: {:.2f}".format(acc_train11))
            print("Testing accuracy: {:.2f}".format(acc_test11))
            print("USER " + str(idx) + " BACKDOOR data accuracy: {:.2f}".format(acc_train11),file=f)
            print("Testing accuracy: {:.2f}".format(acc_test11),file=f)

        # update global weights
        w_glob = FedAvg(x_locals)
        # use m_Krum update global weights



        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # 每轮迭代都测试全局模型准确率
        model_test = copy.deepcopy(net_glob)
        model_test.eval()
        acc_train11, loss_train11 = test_img(model_test, dataset_test, args)

        print("Epoch "+str(iter)+" Global accuracy: {:.2f}".format(acc_train11))
        print("Epoch " + str(iter) + " Global accuracy: {:.2f}".format(acc_train11),file=f)

        # 定义两个数组


        Accuracy_list.append(100 * acc_train11 / (len(dataset_test)))


        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg),file=f)
        loss_train.append(loss_avg)
        # 计算启动时间和结束时间的时间差
        end = time.perf_counter()
        print('运行时间 : %s 秒' % (end - start))
        print('Epoch '+ str(iter) + 'running time : %s s' % (end - start),file=f)

        # 全局模型使用后门数据集嵌入后门
        # global_backdoor = LocalUpdate(args=args, dataset=dataset_backdoor_train, idxs=dict_users[1])
        # w1, loss = global_backdoor.train(net=copy.deepcopy(net_glob).to(args.device))
        # net_glob.load_state_dict(w1)
        # model_global_backdoor = copy.deepcopy(net_glob)
        # # net_glob = copy.deepcopy(model_global_backdoor)
        # # 测试后门准确率
        # model_test1 = copy.deepcopy(net_glob)
        # model_test1.eval()
        # acc_train11, loss_train11 = test_img(model_test1, dataset_backdoor_train, args)
        # print("Epoch " + str(iter) + " Global_backdoor backdoor data accuracy: {:.2f}".format(acc_train11))
        # print("Epoch " + str(iter) + " Global_backdoor backdoor data accuracy: {:.2f}".format(acc_train11), file=f)
        # acc_train11, loss_train11 = test_img(model_test1, dataset_test, args)
        # print("Epoch " + str(iter) + " Global_backdoor test accuracy: {:.2f}".format(acc_train11))
        # print("Epoch " + str(iter) + " Global_backdoor test accuracy: {:.2f}".format(acc_train11), file=f)

    # 我这里迭代了30次，所以x的取值范围为(0，30)，然后再将每次相对应的准确率以及损失率附在x上
    x1 = range(0, 30)
    y1 = Accuracy_list
    plt.ylim([0, 1])
    plt.subplot(1, 1, 1)  #plt.subplot('行','列','编号')
    plt.plot(x1, y1, '.-')
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    # plt.show()
    plt.savefig("accuracy_loss.jpg")

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    # torch.save(net_glob.state_dict(), 'model/mnist_backdoor.pt')
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_backdoor, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("BACKDOOR data accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    print("BACKDOOR data accuracy: {:.2f}".format(acc_train),file=f)
    print("Testing accuracy: {:.2f}".format(acc_test),file=f)

    f.close()  # 关闭文件