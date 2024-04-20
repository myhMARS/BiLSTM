import pprint
import re
import time

import torch
from torch.utils.data import TensorDataset

import gui
from model import Data, Model, Metrics

max_len = 32


def viterbi(nodes, trans, w):
    paths = {'b': nodes[0]['b'], 's': nodes[0]['s']}  # 第一层，只有两个节点
    pprint.pprint(nodes)
    for l in range(1, len(nodes)):  # 后面的每一层
        paths_ = paths.copy()  # 先保存上一层的路径
        paths = {}
        for i in nodes[l].keys():  # i为本层节链接点
            # 对于本层节点，找出最短路径。
            nows = {}
            # 上一层的每个结点到本层节点的连接
            for j in paths_.keys():  # j为上层节点
                if j[-1] + i in trans.keys():  # 若转移概率不为0
                    nows[j + i] = paths_[j] + nodes[l][i] + w * trans[j[-1] + i]
            nows = sorted(nows.items(), key=lambda x: x[1], reverse=True)
            paths[nows[0][0]] = nows[0][1]
    paths = sorted(paths.items(), key=lambda x: x[1], reverse=True)
    return paths[0][0]


def cut_words(text, dic, model, trans, w):
    text = re.split('[，。！？、\n]', text)
    sens = []
    Xs = []
    for sentence in text:
        sen = []
        X = []
        sentence = list(sentence)
        for s in sentence:
            s = s.strip()
            if not s == '' and s in dic.char2id:
                sen.append(s)
                X.append(dic.char2id[s])
        if len(X) > max_len:
            sen = sen[:max_len]
            X = X[:max_len]
        else:
            for i in range(max_len - len(X)):
                X.append(0)

        if len(sen) > 0:
            Xs.append(X)
            sens.append(sen)
    Xs = torch.tensor(Xs, dtype=torch.long).cuda()
    ys = model.predict(Xs)
    results = ''

    for i in range(ys.shape[0]):
        nodes = [dict(zip(['s', 'b', 'm', 'e'], d[:4])) for d in ys[i]]
        ts = viterbi(nodes, trans, w)
        for x in range(len(sens[i])):
            if ts[x] in ['s', 'e']:
                results += sens[i][x] + '/'
            else:
                results += sens[i][x]

    return results[:-1]


def main():
    data = Data(32)
    data.load_vocab('./data/msr_training_words.utf8')
    embedding_size = 128
    hidden_size = 64
    model = Model(data.vocab_size + 1, embedding_size, hidden_size, 5)
    x_train, y_train, trans = data.load_data('./data/msr_training.utf8')
    # print(trans)

    def init_train():
        model.create_model()
        model.dataset = TensorDataset(x_train, y_train)
        model.train(epochs=30, batch_size=128, optimizer='Adam')

        model.save('./models/msr_Adam_default_30.pt')

    def init_load():
        model.load('./models/msr_Adam_default_30.pt')

    def test():
        x_test, y_test, _ = data.load_data('./data/msr_test_gold.utf8')
        y_pred = model.predict(x_test)
        metrics = Metrics(y_pred, y_test, 4)
        metrics.show('./models/msr_Adam_default_30_report.txt')
        # metrics.use_sklearn()

    def segment_text(win):
        start_time = time.time()  # 记录开始时间
        input_text = win.input_text_box.get("1.0", "end-1c")  # 获取输入文本框中的文本
        seg_list = cut_words(input_text, data, model, trans, w=4)  # 使用BiLSTM模型进行分词
        output_text = " ".join(seg_list)  # 将分词结果用空格连接成字符串
        win.output_text_box.delete("1.0", "end")  # 清空输出文本框
        win.output_text_box.insert("1.0", output_text)  # 将分词结果插入输出文本框
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算用时
        win.time_label.config(text=f"分词用时：{elapsed_time:.4f} 秒")  # 更新时间标签的文本

    def show_gui():
        window = gui.GUI()
        window.segment_button.config(command=lambda: segment_text(window))
        window.mainloop()

    # init_train()  # 重新训练模型
    init_load()  # 加载已训练模型
    # print('模型加载完成')
    # model.show()
    # test()
    show_gui()


if __name__ == '__main__':
    main()
