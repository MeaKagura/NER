import torch
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import Dataset


class BertDataset(Dataset):
    def __init__(self, sentence_list, labels_list, config, word_pad_idx=0, label_pad_idx=-1):
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True)
        self.label2id = config.label2id
        self.id2label = {_id: _label for _label, _id in list(config.label2id.items())}
        self.dataset = self.encode(sentence_list, labels_list)
        self.word_pad_idx = word_pad_idx
        self.label_pad_idx = label_pad_idx
        self.device = config.device

    def encode(self, origin_sentence_list, origin_labels_list):
        """
        对token及其标签进行编码，并存储到字典data中
        examples:
            origin_sentence:['浙', '商', '银', '行', '企', '业', '信', '贷', '部']
            intermediate_sentence:['[CLS]', '浙', '商', '银', '行', '企', '业', '信', '贷', '部']
            sentence:([101, 3851, 1555, 7213, 6121, 821, 689, 928, 6587, 6956],
                        array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]))
            origin_labels:['B-company', 'I-company', 'I-company', 'I-company', '0', '0', '0', '0', '0']
            labels:[3, 13, 13, 13, 0, 0, 0, 0, 0]
        """
        dataset = []
        sentence_list = []
        labels_list = []

        # 对token进行编码
        for origin_sentence in origin_sentence_list:
            # we can not use encode_plus because our sentence_list are aligned to labels_list in list type
            sentence = []
            word_lens = []
            # 分词
            for token in origin_sentence:
                sentence.append(self.tokenizer.tokenize(token))
                word_lens.append(len(token))
            # 在单个token的列表开头加上[CLS]
            sentence = ['[CLS]'] + [item for token in sentence for item in token]
            # 将token映射为id
            token_start_idxs = 1 + np.cumsum([0] + word_lens[:-1])
            sentence_list.append((self.tokenizer.convert_tokens_to_ids(sentence), token_start_idxs))

        # 对标签进行编码
        for origin_labels in origin_labels_list:
            labels = [self.label2id.get(origin_label) for origin_label in origin_labels]
            labels_list.append(labels)

        # 将句子与相应的标签打包成元组
        for sentence, labels in zip(sentence_list, labels_list):
            dataset.append((sentence, labels))
        return dataset

    def __getitem__(self, idx):
        """读取数据集的元素"""
        sentence = self.dataset[idx][0]
        labels = self.dataset[idx][1]
        return [sentence, labels]

    def __len__(self):
        """读取数据集的长度"""
        return len(self.dataset)

    def collate_fn(self, batch):
        """
        读取一个batch的数据时，进行以下处理:
            1. padding: 将每个batch的sentence 与 labels padding到同一长度（batch中最长句子的长度）
            2. aligning: 找到每个sentence sequence里面有label项，文本与label对齐
            3. tensor：转化为tensor
        examples:
            origin_sentence:[101, 3851, 1555, 7213, 6121, 821, 689, 928, 6587, 6956]
            sentence:[101, 3851, 1555, 7213, 6121, 821, 689, 928, 6587, 6956, pad_idx, pad_idx]
            origin_label_starts:[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
            label_starts:[ 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
            origin_labels:[3, 13, 13, 13, 0, 0, 0, 0, 0]
            labels:[3, 13, 13, 13, 0, 0, 0, 0, 0, pad_idx, pad_idx]
        """
        sentence_list = [x[0] for x in batch]
        labels_list = [x[1] for x in batch]

        # batch length
        batch_len = len(sentence_list)

        # compute length of longest sentence in batch
        max_len = max([len(sentence[0]) for sentence in sentence_list])
        max_label_len = 0

        # 初始化
        batch_sentences = self.word_pad_idx * np.ones((batch_len, max_len))
        batch_label_starts = []

        # padding and aligning
        for j in range(batch_len):
            # 填充原来的句子
            cur_len = len(sentence_list[j][0])
            batch_sentences[j][:cur_len] = sentence_list[j][0]
            # 原句子内的start填充为1（[CLS]不算），其余填充为0
            label_start_idx = sentence_list[j][-1]
            label_starts = np.zeros(max_len)
            label_starts[[idx for idx in label_start_idx if idx < max_len]] = 1
            batch_label_starts.append(label_starts)
            max_label_len = max(int(sum(label_starts)), max_label_len)

        # 填充标签
        batch_labels = self.label_pad_idx * np.ones((batch_len, max_label_len))
        for j in range(batch_len):
            cur_tags_len = len(labels_list[j])
            batch_labels[j][:cur_tags_len] = labels_list[j]

        # convert data to torch LongTensors
        batch_sentences = torch.tensor(batch_sentences, dtype=torch.long)
        batch_label_starts = torch.tensor(batch_label_starts, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        # shift tensors to GPU if available
        batch_sentences, batch_label_starts = batch_sentences.to(self.device), batch_label_starts.to(self.device)
        batch_labels = batch_labels.to(self.device)
        return [batch_sentences, batch_label_starts, batch_labels]
