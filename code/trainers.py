import numpy as np
import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam

from utils import recall_at_k, ndcg_k, get_metric


class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        # self.criterion = nn.BCELoss()
        self.criterion = nn.CrossEntropyLoss(reduce=False)

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids):
        # [batch, seq_len, hidden_size]
        item_emb = self.model.item_embeddings.weight
        logits = torch.matmul(seq_out.view(-1, seq_out.size(-1)), item_emb.transpose(0, 1))
        istarget = (pos_ids > 0).view(-1).float()
        loss = torch.sum(self.criterion(logits, pos_ids.view(-1)) * istarget)/torch.sum(istarget)
        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

class FinetuneTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(FinetuneTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

    def iteration(self, epoch, dataloader, full_sort=False, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                _, input_ids, target_pos, target_neg, _ = batch
                # Binary cross_entropy
                sequence_output = self.model.finetune(input_ids)
                loss = self.cross_entropy(sequence_output, target_pos)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.model.finetune(input_ids)

                    recommend_output = recommend_output[:, -1, :]
                    # 推荐的结果

                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition 时间复杂度O(n)  argsort O(nlogn) 只会做
                    # 加负号"-"表示取大的值
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    # 根据返回的下标 从对应维度分别取对应的值 得到每行topk的子表
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    # 对子表进行排序 得到从大到小的顺序
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    # 再取一次 从ind中取回 原来的下标
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list)


class MVSTrainer(Trainer):

    def __init__(self, model, model_gen,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(MVSTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

        self.model_gen = model_gen
        if self.cuda_condition:
            self.model_gen.cuda()
        self.criterion = nn.BCELoss()
        self.ce_criterion = nn.CrossEntropyLoss(reduce=False)
        self.kl_criterion = nn.KLDivLoss(reduce=False)
        self.variance = args.variance

    def VDA(self, seq_out_prob, ori_embeddings, ori_embeddings_prob):
        # Generate the Distribution
        batch, len, hidden = seq_out_prob.size()
        seq_out_prob = seq_out_prob.view(batch * len, -1)
        embed_matrix = self.model_gen.item_embeddings.weight
        score = torch.softmax(torch.matmul(seq_out_prob, embed_matrix.transpose(0, 1)), -1)  # [B*L, V]
        noise = torch.randn(score.size(), device=score.device) * self.variance
        score = score + noise

        # Aggregate the Dis Inputs
        new_embeddings = torch.matmul(score, embed_matrix).view(batch, len, -1)  # [B, L, hidden]
        fused_embeddings_prob = torch.cat([ori_embeddings_prob[:, :1, :], new_embeddings[:, :-1, :]], 1)
        # mixed_embeddings_prob = 0.5*fused_embeddings_prob + 0.5*ori_embeddings_prob

        # Aggregate the Gen Inputs
        embed_matrix = self.model.item_embeddings.weight
        new_embeddings = torch.matmul(score, embed_matrix).view(batch, len, -1)  # [B, L, hidden]
        fused_embeddings = torch.cat([ori_embeddings[:, :1, :], new_embeddings[:, :-1, :]], 1)
        # mixed_embeddings = 0.5*fused_embeddings + 0.5*ori_embeddings
        # return mixed_embeddings, mixed_embeddings_prob
        return fused_embeddings, fused_embeddings_prob

    def VDA_FreeLB(self, input_ids, target_pos, seq_out, seq_out_prob, ori_embeddings, ori_embeddings_prob):
        # Generate the Distribution
        batch, len, hidden = seq_out_prob.size()
        seq_out_prob = seq_out_prob.view(batch * len, -1)
        embed_matrix = self.model_gen.item_embeddings.weight
        score = torch.softmax(torch.matmul(seq_out_prob, embed_matrix.transpose(0, 1)), -1)  # [B*L, V]
        noise = torch.randn(score.size(), device=score.device) * self.variance
        noise = torch.clamp(noise, -1.0 * self.variance, self.variance)
        noise.requires_grad = True
        score_new = score + noise

        for _ in range(1):
            # Aggregate the Dis Inputs
            new_embeddings = torch.matmul(score_new, embed_matrix).view(batch, len, -1)  # [B, L, hidden]
            fused_embeddings_prob = torch.cat([ori_embeddings_prob[:, :1, :], new_embeddings[:, :-1, :]], 1)
            # Aggregate the Gen Inputs
            embed_matrix = self.model.item_embeddings.weight
            new_embeddings = torch.matmul(score_new, embed_matrix).view(batch, len, -1)  # [B, L, hidden]
            fused_embeddings = torch.cat([ori_embeddings[:, :1, :], new_embeddings[:, :-1, :]], 1)

            VDA_embeddings, VDA_embeddings_prob = fused_embeddings, fused_embeddings_prob
            # seq_output_prob_VDA = self.model_gen.finetune_emb(input_ids, VDA_embeddings_prob)
            seq_output_VDA = self.model.finetune_emb(input_ids, VDA_embeddings)
            _, kl_loss_vda = self.cross_entropy_for_KD(seq_output_VDA, seq_out, target_pos, is_vda=True)

            noise_grad = torch.autograd.grad(kl_loss_vda, noise)[0]
            noise = noise + (noise_grad / torch.norm(noise_grad)).mul_(0.1)
            noise = torch.clamp(noise, -1.0 * self.variance, self.variance)
            score_new = score + noise

        # Aggregate the Dis Inputs
        new_embeddings = torch.matmul(score_new, embed_matrix).view(batch, len, -1)  # [B, L, hidden]
        fused_embeddings_prob = torch.cat([ori_embeddings_prob[:, :1, :], new_embeddings[:, :-1, :]], 1)
        # Aggregate the Gen Inputs
        embed_matrix = self.model.item_embeddings.weight
        new_embeddings = torch.matmul(score_new, embed_matrix).view(batch, len, -1)  # [B, L, hidden]
        fused_embeddings = torch.cat([ori_embeddings[:, :1, :], new_embeddings[:, :-1, :]], 1)

        return fused_embeddings, fused_embeddings_prob

    def cross_entropy_for_KD(self, seq_out, seq_out_prob, pos_ids, is_vda=True):
        # [batch seq_len hidden_size]
        batch, len, hidden = seq_out.size()
        seq_out = seq_out.view(batch * len, -1)
        embed_matrix = self.model.item_embeddings.weight
        score = torch.matmul(seq_out, embed_matrix.transpose(0, 1))  # [B*L, V]

        if is_vda == False:
            embed_matrix = self.model_gen.item_embeddings.weight
        seq_out_prob = seq_out_prob.view(batch * len, -1)
        origin_score = torch.matmul(seq_out_prob, embed_matrix.transpose(0, 1))  # [B*L, V]

        n_loss = 0.5 * self.kl_criterion(torch.log_softmax(score, -1), torch.softmax(origin_score, -1)) + \
                 0.5 * self.kl_criterion(torch.log_softmax(origin_score, -1), torch.softmax(score, -1))
        n_loss = torch.sum(n_loss, -1)
        n_mask = (pos_ids != 0).float()  # *(masked_item_sequence != self.args.mask_id).float()
        n_loss = torch.sum(n_loss * n_mask.flatten()) / torch.sum(n_mask)

        mip_loss = self.ce_criterion(score, pos_ids.view(-1))
        mip_mask = (pos_ids != 0).float()  # *(masked_item_sequence != self.args.mask_id).float()
        mip_loss = torch.sum(mip_loss * mip_mask.flatten()) / torch.sum(mip_mask)

        return mip_loss, n_loss

    def iteration(self, epoch, dataloader, full_sort=False, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()
            self.model_gen.eval()
            rec_ce_loss = 0.0
            rec_kl_loss = 0.0
            rec_ce_vda_loss = 0.0
            rec_kl_vda_loss = 0.0

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                _, input_ids, target_pos, target_neg, _ = batch
                # Binary cross_entropy
                seq_output_prob = self.model_gen.finetune(input_ids)
                ori_embeddings_prob = self.model_gen.item_embeddings(input_ids)
                seq_output = self.model.finetune(input_ids)
                ori_embeddings = self.model.item_embeddings(input_ids)
                loss, kl_loss = self.cross_entropy_for_KD(seq_output, seq_output_prob, target_pos, is_vda=True)
                VDA_embeddings_1, VDA_embeddings_prob_1 = self.VDA_FreeLB(input_ids, target_pos, seq_output,
                                                                          seq_output_prob,
                                                                          ori_embeddings, ori_embeddings_prob)
                VDA_embeddings_2, VDA_embeddings_prob_2 = self.VDA_FreeLB(input_ids, target_pos, seq_output,
                                                                          seq_output_prob,
                                                                          ori_embeddings, ori_embeddings_prob)
                # VDA_embeddings_3, VDA_embeddings_prob_3 = self.VDA_FreeLB(input_ids, target_pos, seq_output,
                #                                                           seq_output_prob,
                #                                                           ori_embeddings, ori_embeddings_prob)
                # seq_output_prob_VDA = self.model_gen.finetune_emb(input_ids, VDA_embeddings_prob)
                seq_output_VDA_1 = self.model.finetune_emb(input_ids, VDA_embeddings_1)
                seq_output_VDA_2 = self.model.finetune_emb(input_ids, VDA_embeddings_2)
                # seq_output_VDA_3 = self.model.finetune_emb(input_ids, VDA_embeddings_3)

                loss_vda_1, kl_loss_vda_1 = self.cross_entropy_for_KD(seq_output_VDA_1, seq_output, target_pos, is_vda=True)
                loss_vda_2, kl_loss_vda_2 = self.cross_entropy_for_KD(seq_output_VDA_2, seq_output, target_pos, is_vda=True)
                _, kl_loss_vda_cross = self.cross_entropy_for_KD(seq_output_VDA_1, seq_output_VDA_2, target_pos,
                                                                 is_vda=True)

                # _, kl_loss_vda_da = self.cross_entropy(seq_output_VDA, seq_output_prob_VDA, target_pos, is_vda=False)

                total_loss = loss + \
                             self.args.loss_1*kl_loss + \
                             self.args.loss_2*0.2 / 2 * (loss_vda_1 + loss_vda_2) + \
                             self.args.loss_3*1 / 3 * (kl_loss_vda_1 + kl_loss_vda_2 + kl_loss_vda_cross)

                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()

                rec_ce_loss += loss.item()
                rec_kl_loss += kl_loss.item()
                rec_ce_vda_loss += loss_vda_1.item() + loss_vda_2.item()
                rec_kl_vda_loss += kl_loss_vda_1.item() + kl_loss_vda_2.item() + kl_loss_vda_cross.item()

            post_fix = {
                "epoch": epoch,
                "rec_ce_loss": '{:.4f}'.format(rec_ce_loss / len(rec_data_iter)),
                "rec_kl_loss": '{:.4f}'.format(rec_kl_loss / len(rec_data_iter)),
                "rec_ce_vda_loss": '{:.4f}'.format(rec_ce_vda_loss / len(rec_data_iter)),
                "rec_kl_vda_loss": '{:.4f}'.format(rec_kl_vda_loss / len(rec_data_iter)),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.model.finetune(input_ids)

                    recommend_output = recommend_output[:, -1, :]
                    # 推荐的结果

                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition 时间复杂度O(n)  argsort O(nlogn) 只会做
                    # 加负号"-"表示取大的值
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    # 根据返回的下标 从对应维度分别取对应的值 得到每行topk的子表
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    # 对子表进行排序 得到从大到小的顺序
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    # 再取一次 从ind中取回 原来的下标
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list)
