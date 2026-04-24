import sys
import torch
from load import *
from ultis import *
import time
import random
from torch import optim
import torch.utils.data as data
from tqdm import tqdm
from models import *
from torchstat import stat
from torchsummary import summary
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
import datetime
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

torch.set_num_threads(1)

def calculate_MRR(prob, label):
    """Mean Reciprocal Rank: Reciprocal of the rank of the first relevant item """

    rlt = 0
    prob=prob.data.cpu().numpy()
    label=label.data.cpu().numpy()
    for y_true, y_pred in zip( label,prob):
        rec_list = y_pred.argsort()[::-1]
        if y_true == -1:
            pass
        else:
            r_idx = np.where(rec_list == y_true)[0]
            rlt += 1 / (r_idx + 1)
    return rlt

def calculate_acc(prob, label):
    acc_train = [0.00, 0.00, 0.00, 0.00,0.00]
    for i, k in enumerate([1, 2, 5, 10, 20]):
        # topk_batch (N, k)
        _, topk_predict_batch = torch.topk(prob, k=k)
        for j, topk_predict in enumerate(to_npy(topk_predict_batch)):
            if to_npy(label)[j] in topk_predict:
                acc_train[i] += 1

    return np.array(acc_train)

def sampling_prob(prob, label, num_neg):

    num_label, l_m = prob.shape[0], prob.shape[1]-1  # prob (N, L)
    label = label.view(-1)  # label (N)
    init_label = np.linspace(0, num_label-1, num_label)  # (N), [0 -- num_label-1]
    init_prob = torch.zeros(size=(num_label, num_neg+len(label)))  # (N, num_neg+num_label)


    random_ig = random.sample(range(1, l_m+1), num_neg)  # (num_neg) from (1 -- l_max)
    while len([lab for lab in label if lab in random_ig]) != 0:  # no intersection
        random_ig = random.sample(range(1, l_m+1), num_neg)

    global global_seed
    random.seed(global_seed)
    global_seed += 1

    # place the pos labels ahead and neg samples in the end
    for k in range(num_label):
        for i in range(num_neg + len(label)):
            if i < len(label):
                init_prob[k, i] = prob[k, label[i]]
            else:
                init_prob[k, i] = prob[k, random_ig[i-len(label)]]

    return torch.FloatTensor(init_prob), torch.LongTensor(init_label)  # (N, num_neg+num_label), (N)


class DataSet(data.Dataset):
    def __init__(self, traj, m1, v, label, length):
        # (NUM, M, 3), (NUM, M, M, 2), (L, L), (NUM, M), (NUM), (NUM)
        self.traj, self.mat1, self.vec, self.label, self.length = traj, m1, v, label, length

    def __getitem__(self, index):
        traj = self.traj[index].to(device)
        mats1 = self.mat1[index].to(device)
        vector = self.vec[index].to(device)
        label = self.label[index].to(device)
        length = self.length[index].to(device)
        return traj, mats1, vector, label, length

    def __len__(self):  # no use
        return len(self.traj)

torch.multiprocessing.set_start_method('spawn')

class Trainer:
    def __init__(self, model, record):
        # load other parameters
        self.model = model.to(device)
        self.records = record
        self.start_epoch = record['epoch'][-1] if load else 1
        self.num_neg = 10
        self.interval = 1000
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.num_epoch = 100
        self.threshold = np.mean(record['acc_valid'][-1]) if load else 0  # 0 if not update

        # (NUM, M, 3), (NUM, M, M, 2), (L, L), (NUM, M, M), (NUM, M), (NUM) i.e. [*M]
        self.traj, self.mat1, self.mat2s, self.mat2t, self.label, self.len = \
            trajs, mat1, mat2s, mat2t, labels, lens


        # nn.cross_entropy_loss counts target from 0 to C - 1, so we minus 1 here.
        self.dataset = DataSet(self.traj, self.mat1, self.mat2t, self.label-1, self.len)
        self.data_loader = data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False,num_workers=0,drop_last=True)

    def train(self):
        # set optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1)


        for t in range(self.num_epoch):
            # settings or validation and test
            valid_size, test_size = 0, 0
            acc_valid, acc_test = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]

            MRR_valid, MRR_test = 0, 0

            bar = tqdm(total=part)
            for step, item in enumerate(self.data_loader):
                # get batch data, (N, M, 3), (N, M, M, 2), (N, M, M), (N, M), (N)
                person_input, person_m1, person_m2t, person_label, person_traj_len = item
                # print(person_input.shape)
                person_traj_len[-1]=person_traj_len[-1] + 2

                input_mask = torch.zeros((self.batch_size, max_len, 3), dtype=torch.long).to(device)
                m1_mask = torch.zeros((self.batch_size, max_len, max_len, 2), dtype=torch.float32).to(device)
                for mask_len in range(1, person_traj_len[-1]+1):  # from 1 -> len

                    # if mask_len != person_traj_len[0]:
                    #     continue
                    input_mask[:, :mask_len] = 1.
                    m1_mask[:, :mask_len, :mask_len] = 1.

                    train_input = person_input * input_mask
                    train_m1 = person_m1 * m1_mask
                    train_m2t = person_m2t[:, mask_len - 1]
                    train_label = person_label[:, mask_len - 1]  # (N)
                    train_len = torch.zeros(size=(self.batch_size,), dtype=torch.long).to(device) + mask_len
                    traj_t = (person_input[:, :, 2] - 1) % hours + 1  # segment time by 24 hours * 7 days

                    prob = self.model(train_input, train_m1, self.mat2s, train_m2t, train_len,
                                      A, raw_X,
                                     # mat2s_L, mat2s_S
                                      )  # (N, L)

                    if mask_len <= person_traj_len[0] - 2:  # only training
                        # nn.utils.clip_grad_norm_(self.model.parameters(), 10)

                        if (self.batch_size!= 1):
                            prob=prob.squeeze()

                        prob_sample, label_sample = sampling_prob(prob, train_label, self.num_neg)
                        criterion_time = maksed_mse_loss

                        loss_train = F.cross_entropy(prob_sample, label_sample)

                        #ALL_loss.backward()
                        loss_train.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()

                    elif mask_len == person_traj_len[0] - 1:  # only validation
                        valid_size += person_input.shape[0]
                        acc_valid += calculate_acc(prob, train_label)
                        MRR_valid += calculate_MRR(prob, train_label)
                    elif mask_len == person_traj_len[0]:  # only test
                        print('test')
                        test_size += person_input.shape[0]
                        acc_test += calculate_acc(prob, train_label)
                        MRR_test += calculate_MRR(prob, train_label)

                bar.update(self.batch_size)
            bar.close()

            acc_valid = np.array(acc_valid) / valid_size
            print('valid_size',valid_size)
            print('epoch:{}, time:{}, valid_acc:{}'.format(self.start_epoch + t, time.time() - start, acc_valid))
            print('valid_MRR:', MRR_valid / valid_size)

            acc_test = np.array(acc_test) / test_size
            print('test_size', test_size)
            print('epoch:{}, time:{}, test_acc:{}'.format(self.start_epoch + t, time.time() - start, acc_test))
            print('test_MRR:', MRR_test / test_size)

            self.records['acc_valid'].append(acc_valid)
            self.records['acc_test'].append(acc_test)
            self.records['epoch'].append(self.start_epoch + t)

            if self.threshold < np.mean(acc_valid):
                self.threshold = np.mean(acc_valid)
                # save the model
                torch.save({'state_dict': self.model.state_dict(),
                            'records': self.records,
                            'time': time.time() - start},
                           'best_stan_win_1000_' + dname + '.pth')

            with open('./results/'+str(dname)+'/'+'ALB/'+dname+'_'+rlabel+'.txt', 'a+') as writers:  # 打开文件
                tttt = 'epoch:{}, time:{}'.format(self.start_epoch + t, time.time() - start)
                va = 'valid_acc:{}'.format(acc_valid)
                b =  np.array(MRR_valid) / valid_size
                vr = 'valid_MRR:{}'.format(b)
                ta = 'test_acc:{}'.format(acc_test)
                c = np.array(MRR_test) / test_size
                tr = 'test_MRR:{}'.format(c)
                mean_t = np.mean(acc_test)
                mean_v = np.mean(acc_valid)
                mean_tt = 'mean_test:{}'.format(mean_t)
                mean_vv = 'mean_val:{}'.format(mean_v)

                writers.write(tttt + '\n')
                writers.write(va + '\n')
                writers.write(mean_vv + '\n')
                writers.write(ta + '\n')
                writers.write(mean_tt + '\n')
                writers.write(vr + '\n')
                writers.write(tr + '\n')
                writers.write('\n')



    def inference(self):
        user_ids = []
        for t in range(self.num_epoch):
            # settings or validation and test
            valid_size, test_size = 0, 0
            acc_valid, acc_test = [0, 0, 0, 0], [0, 0, 0, 0]
            cum_valid, cum_test = [0, 0, 0, 0], [0, 0, 0, 0]
            MRR_valid, MRR_test = [0], [0]
            for step, item in enumerate(self.data_loader):
                # get batch data, (N, M, 3), (N, M, M, 2), (N, M, M), (N, M), (N)
                person_input, person_m1, person_m2t, person_label, person_traj_len = item

                input_mask = torch.zeros((self.batch_size, max_len, 3), dtype=torch.long).to(device)
                m1_mask = torch.zeros((self.batch_size, max_len, max_len, 2), dtype=torch.float32).to(device)
                for mask_len in range(1, person_traj_len[0] + 1):  # from 1 -> len
                    input_mask[:, :mask_len] = 1.
                    m1_mask[:, :mask_len, :mask_len] = 1.

                    train_input = person_input * input_mask
                    train_m1 = person_m1 * m1_mask
                    train_m2t = person_m2t[:, mask_len - 1]
                    train_label = person_label[:, mask_len - 1]  # (N)
                    train_len = torch.zeros(size=(self.batch_size,), dtype=torch.long).to(device) + mask_len

                    prob = self.model(train_input, train_m1, self.mat2s, train_m2t, train_len)  # (N, L)


                    if mask_len <= person_traj_len[0] - 2:  # only training
                        continue

                    elif mask_len == person_traj_len[0] - 1:  # only validation
                        acc_valid = calculate_acc(prob, train_label)
                        cum_valid += calculate_acc(prob, train_label)
                        MRR_valid += calculate_MRR(prob, train_label)

                    elif mask_len == person_traj_len[0]:  # only test
                        acc_test = calculate_acc(prob, train_label)
                        cum_test += calculate_acc(prob, train_label)
                        MRR_test += calculate_MRR(prob, train_label)

                print(step, acc_valid, acc_test)

                if acc_valid.sum() == 0 and acc_test.sum() == 0:
                    user_ids.append(step)


if __name__ == '__main__':

    # load data
    dname = 'NYC'
    # dname = 'TKY'
    file = open('./data/' + dname + '_data.pkl', 'rb')
    file_data = joblib.load(file)
    # tensor(NUM, M, 3), np(NUM, M, M, 2), np(L, L), np(NUM, M, M), tensor(NUM, M), np(NUM)
    [trajs, mat1, mat2s, mat2t, labels, lens, u_max, l_max] = file_data
    print(u_max, l_max)
    mat1, mat2s, mat2t, lens = torch.FloatTensor(mat1), torch.FloatTensor(mat2s).to(device), \
        torch.FloatTensor(mat2t), torch.LongTensor(lens)

    # Build POI graph (built from train_df)
    print('Loading POI graph...')
    raw_A = load_graph_adj_mtx('data_graph/' + dname + '_graph/graph_A.npy')
    raw_X = load_graph_node_features('data_graph/' + dname + '_graph/graph_X.csv',
                                     'checkin_cnt',
                                     'poi_index',
                                     'poi_deltat',
                                     'latitude',
                                     'longitude')
    if isinstance(raw_X, np.ndarray):
        raw_X = torch.from_numpy(raw_X)
    raw_X = raw_X.to(device, dtype=torch.float)
    A = calculate_laplacian_matrix(raw_A, mat_type='hat_rw_normd_lap_mat')
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A)
    A = A.to(device, dtype=torch.float)
    print("complete calculate A")

    part = 1083
    start = 0

    if ((start + part) <= u_max):
        trajs, mat1, mat2t, labels, lens = \
            trajs[start:start + part], mat1[start:start + part], mat2t[start:start + part], labels[
                                                                                            start:start + part], lens[
                                                                                                                 start:start + part]
    else:
        trajs, mat1, mat2t, labels, lens = \
            trajs[start:-1], mat1[start:-1], mat2t[start:-1], labels[start:-1], lens[start:-1]

    print("prepare for model finished")

    ex = mat1[:, :, :, 0].max(), mat1[:, :, :, 0].min(), mat1[:, :, :, 1].max(), mat1[:, :, :, 1].min()
    # print(ex)

    dim = dim
    rlabel = str(start) + "_" + str(start + part) + str(dim)+'_all'
    # rlabel = str(start) + "_" + str(start + part) + 's_test'
    print('rlabel=', rlabel)
    # w_ag是poi整体注意分数，w_poi是对于二跳分布的注意分数
    hp = embed_dim, w_ag, w_poi, nhid, ninput, noutput, dropout_GCN, Node_nhid = dim, 0.6, 1, [16,
                                                                                               32], dim, dim, 0.3, 100  # 0.6  0.4
    # hp = embed_dim, w_ag, w_poi, nhid, ninput, noutput, dropout_GCN = 50, 0.6, 0.4, [16, 32], 50,50, 0.3
    with open('./results/' + str(dname) + '/' + 'ALB/' + dname + '_' + rlabel + '.txt', 'a+',
              encoding='utf-8') as writers:
        writers.write(str(datetime.datetime.now()) + '\n')
        writers.write(str(hp) + '\n')
        writers.write(str(start) + str(part) + '\n')
        # writers.truncate(0)
    print('embed_dim, w_ag, w_poi, nhid, ninput, noutput, dropout_GCN=' + str(hp) + '\n')


    stan = Model(t_dim=hours + 1, l_dim=l_max + 1, u_dim=u_max + 1, ex=ex, hp=hp)
    num_params = 0

    for name in stan.state_dict():
        print(name)

    def model_structure(model):
        blank = ' '
        print('-' * 90)
        print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
              + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
              + ' ' * 3 + 'number' + ' ' * 3 + '|')
        print('-' * 90)
        num_para = 0
        type_size = 1  # 如果是浮点数就是4

        for index, (key, w_variable) in enumerate(model.named_parameters()):
            if len(key) <= 30:
                key = key + (30 - len(key)) * blank
            shape = str(w_variable.shape)
            if len(shape) <= 40:
                shape = shape + (40 - len(shape)) * blank
            each_para = 1
            for k in w_variable.shape:
                each_para *= k
            num_para += each_para
            str_num = str(each_para)
            if len(str_num) <= 10:
                str_num = str_num + (10 - len(str_num)) * blank

            print('| {} | {} | {} |'.format(key, shape, str_num))
        print('-' * 90)
        print('The total number of parameters: ' + str(num_para))
        print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
        print('-' * 90)


    model_structure(stan)

    for param in stan.parameters():
        num_params += param.numel()
    print('num of params', num_params)

    load = False

    if load:
        checkpoint = torch.load('best_stan_win_' + dname + '.pth')
        stan.load_state_dict(checkpoint['state_dict'])
        start = time.time() - checkpoint['time']
        records = checkpoint['records']
    else:
        records = {'epoch': [], 'acc_valid': [], 'acc_test': []}
        start = time.time()

    trainer = Trainer(stan, records)
    trainer.train()
    # trainer.inference()
