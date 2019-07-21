import numpy as np
import pandas as pd
import paddle.fluid as fluid

from sklearn.model_selection import train_test_split

from codes.preprocessing import image_to_matrix

image_size = 120


def convolutional_neural_network(img):
    """
    定义卷积神经网络分类器：
        输入的二维图像，经过两个卷积-池化层，一个全连接层，使用以softmax为激活函数的输出层

    Return:
        prediction -- 分类的结果
    """
    # 第一个卷积-池化层
    # 使用32个5*5的滤波器，池化大小为3，池化步长为2，激活函数为Relu
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=32,
        pool_size=3,
        pool_stride=2,
        act='relu')
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    # 第二个卷积-池化层
    # 使用64个5*5的滤波器，池化大小为3，池化步长为2，激活函数为Relu
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=64,
        pool_size=3,
        pool_stride=2,
        act='relu')
    # 全连接层，输出为512维特征
    fc1 = fluid.layers.fc(input=conv_pool_2, size=512, act=None)
    # 以softmax为激活函数的全连接输出层，输出层的大小必须为类别数7
    prediction = fluid.layers.fc(input=fc1, size=7, act='softmax')
    return prediction


def train_program(img, label):
    """
    配置train_program

    Return:
        prediction -- 分类的结果
        avg_cost -- 平均损失
        acc -- 分类的准确率
    """
    prediction = convolutional_neural_network(img)  # 使用自定义的卷积神经网络
    # 使用类交叉熵函数计算predict和label之间的损失函数
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    # 计算平均损失
    avg_cost = fluid.layers.mean(cost)
    # 计算分类准确率
    acc = fluid.layers.accuracy(input=prediction, label=label)
    return prediction, [avg_cost, acc]


# 训练过程中测试模型的好坏
def train_test(test_program, feeder, test_data, batch_size):
    # 将分类准确率存储在acc_set中
    acc_set = []
    # 将平均损失存储在avg_loss_set中
    avg_loss_set = []
    # 将测试集 yield 出的每一个数据传入网络中进行训练
    for data in generator_batches(test_data, batch_size):
        acc_np, avg_loss_np = exe.run(
            program=test_program,
            feed=feeder.feed(data),
            fetch_list=[acc, avg_loss])
        acc_set.append(float(acc_np))
        avg_loss_set.append(float(avg_loss_np))
    # 获得测试数据上的准确率和损失值
    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()
    # 返回平均损失值，平均准确率
    return avg_loss_val_mean, acc_val_mean


def predict(save_dirname='../model/cnn_paddle.model',
            image_file=r'M:\Users\jack\Desktop\C4\CK+DB\cohn-kanade-images\S005\001\S005_001_00000001.png'):
    """
    加载保存的模型进行预测
    :param save_dirname: 模型保存的路径
    :param image_file: 要预测的表情图片路径
    :return:
    """
    img_data = image_to_matrix(image_file)
    # 转为输入所需的大小
    img_data = img_data.reshape((1, 1, image_size, image_size)).astype(np.float32)
    use_cuda = True  # 如想使用GPU，请设置为 True
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    # 创建执行器
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    inference_scope = fluid.core.Scope()

    with fluid.scope_guard(inference_scope):
        # 使用 fluid.io.load_inference_model 获取 inference program desc,
        # feed_target_names 用于指定需要传入网络的变量名
        # fetch_targets 指定希望从网络中fetch出的变量名
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(
            save_dirname, exe, None, None)

        # 将feed构建成字典 {feed_target_name: feed_target_data}
        # 结果将包含一个与fetch_targets对应的数据列表
        results = exe.run(inference_program,
                          feed={feed_target_names[0]: img_data},
                          fetch_list=fetch_targets)
        lab = np.argsort(results)
        # 打印图片的预测结果，结果 +1 转为最初的 1-7
        print('Inference result of ' + image_file + ' is: %d' % (lab[0][0][-1] + 1))


def generator_batches(x, bsize=50):
    """
    generator batch data from x
    :param x: list x, like [[train_data, train_label], ...]
    :param bsize: batch size
    :return: generator batch data
    """
    n = len(x) // bsize
    x = x[:n * bsize]
    for i in range(0, len(x), bsize):
        yield x[i:i + bsize]


# 读取pkl文件的数据
def read_data(file_name):
    data = pd.read_pickle(file_name)
    # 随机打乱后划分数据集，测试集占0.2
    train_data, test_data = train_test_split(data, shuffle=True, test_size=0.2, random_state=42)
    return train_data, test_data


if __name__ == '__main__':

    pkl_file = '../data/data_label_list_120.pkl'
    train_data, test_data = read_data(pkl_file)

    # 一个minibatch中有64个数据
    batch_size = 52
    # 该模型运行在CPU上
    use_cuda = True  # 如想使用GPU，请设置为 True
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    # 创建执行器
    exe = fluid.Executor(place)

    # 输入的原始图像数据，大小为120*120*1
    img = fluid.layers.data(name='img', shape=[1, image_size, image_size], dtype='float32')
    # 标签层，名称为label,对应输入图片的类别标签
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    # 告知网络传入的数据分为两部分，第一部分是img值，第二部分是label值
    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)

    # 调用train_program 获取预测值，损失值，
    prediction, [avg_loss, acc] = train_program(img, label)
    # 选择Adam优化器
    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(avg_loss)
    # 训练的轮数
    epochs = 10
    # 将模型参数存储在名为 save_dirname 的文件中
    save_dirname = '../model/cnn_paddle.model'
    # 设置 main_program 和 test_program
    main_program = fluid.default_main_program()
    test_program = fluid.default_main_program().clone(for_test=True)

    exe.run(fluid.default_startup_program())

    # 开始训练
    lists = []
    step = 0
    for epoch_id in range(0, epochs):
        for data in generator_batches(train_data, batch_size):
            metrics = exe.run(main_program,
                              feed=feeder.feed(data),
                              fetch_list=[avg_loss, acc])
            if step % 50 == 0:  # 每训练50次 打印一次log
                print("Pass %d, Batch %d, Cost %f" % (step, epoch_id, metrics[0]))
            step += 1

        # 测试每个epoch的分类效果
        avg_loss_val, acc_val = train_test(test_program, feeder, test_data, batch_size)
        print("Test with Epoch %d, avg_cost: %s, acc: %s" % (epoch_id, avg_loss_val, acc_val))
        lists.append((epoch_id, avg_loss_val, acc_val))

        # 保存训练好的模型参数用于预测
        if save_dirname is not None:
            fluid.io.save_inference_model(save_dirname,
                                          ['img'], [prediction], exe,
                                          model_filename=None,
                                          params_filename=None)

    # 选择效果最好的pass
    best = sorted(lists, key=lambda lt: float(lt[1]))[0]
    print('Best pass is %s, testing Avgcost is %s' % (best[0], best[1]))
    print('The classification accuracy is %.2f%%' % (float(best[2]) * 100))

    # 训练完成后可进行预测
    # predict()

    print('\n--------------------------Program Finished---------------------------\n')
