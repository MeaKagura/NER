import utils
import config
import logging
import numpy as np
from data_process import DataProcessor
from data_loader import BertDataset
from model import BertNER
from train import train, evaluate

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW

import warnings

warnings.filterwarnings('ignore')


def dev_split(dataset_dir):
    """划分验证集"""
    data = np.load(dataset_dir, allow_pickle=True)
    sentences = data["sentences"]
    labels = data["labels"]
    x_train, x_dev, y_train, y_dev = train_test_split(sentences, labels, test_size=config.dev_split_size, random_state=0)
    return x_train, x_dev, y_train, y_dev


def load_data(mode):
    """加载数据集"""
    if mode == 'train':
        # 分离出验证集
        sentences_train, sentences_dev, labels_train, labels_dev = dev_split(config.train_dir)
    elif mode == 'test':
        train_data = np.load(config.train_dir, allow_pickle=True)
        dev_data = np.load(config.test_dir, allow_pickle=True)
        sentences_train = train_data["sentences"]
        labels_train = train_data["labels"]
        sentences_dev = dev_data["sentences"]
        labels_dev = dev_data["labels"]
    else:
        sentences_train = None
        labels_train = None
        sentences_dev = None
        labels_dev = None
    return sentences_train, sentences_dev, labels_train, labels_dev


def main():
    """train the ner_model"""
    # set the logger
    utils.set_logger(config.log_dir)
    logging.info("device: {}".format(config.device))
    # 处理数据，分离文本和标签
    processor = DataProcessor(config)
    processor.process()
    logging.info("--------raw data process completed.--------")
    # 分离出验证集
    sentences_train, sentences_dev, labels_train, labels_dev = load_data('train')
    # build dataset
    train_dataset = BertDataset(sentences_train, labels_train, config)
    dev_dataset = BertDataset(sentences_dev, labels_dev, config)
    logging.info("--------dataset build completed.--------")
    # get dataset size
    train_size = len(train_dataset)
    # build data_loader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                            shuffle=True, collate_fn=dev_dataset.collate_fn)
    logging.info("--------dataset loader build completed.--------")
    # Prepare ner_model
    device = config.device
    model = BertNER.from_pretrained(config.bert_model, num_labels=len(config.label2id))
    model.to(device)
    # Prepare optimizer
    if config.full_fine_tuning:
        # ner_model.named_parameters(): [bert, classifier, crf]
        bert_optimizer = list(model.bert.named_parameters())
        classifier_optimizer = list(model.classifier.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
            {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 5, 'weight_decay': 0.0},
            {'params': model.crf.parameters(), 'lr': config.learning_rate * 5}
        ]
    # only fine-tune the head classifier
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, correct_bias=False)
    train_steps_per_epoch = train_size // config.batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=(config.epoch_num // 10) * train_steps_per_epoch,
                                                num_training_steps=config.epoch_num * train_steps_per_epoch)

    # Train the ner_model
    logging.info("--------Start Training!--------")
    train(train_loader, dev_loader, model, optimizer, scheduler, config.model_dir)


def test():
    """进行测试"""
    data = np.load(config.test_dir, allow_pickle=True)
    word_test = data["words"]
    label_test = data["labels"]
    test_dataset = BertDataset(word_test, label_test, config)
    logging.info("--------Dataset Build!--------")
    # build data_loader
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, collate_fn=test_dataset.collate_fn)
    logging.info("--------Get Data-loader!--------")
    # Prepare ner_model
    if config.model_dir is not None:
        model = BertNER.from_pretrained(config.model_dir)
        model.to(config.device)
        logging.info("--------Load ner_model from {}--------".format(config.model_dir))
    else:
        logging.info("--------No ner_model to test !--------")
        return
    val_metrics = evaluate(test_loader, model, mode='test')
    val_f1 = val_metrics['f1']
    logging.info("test loss: {}, f1 score: {}".format(val_metrics['loss'], val_f1))
    val_f1_labels = val_metrics['f1_labels']
    for label in config.labels:
        logging.info("f1 score of {}: {}".format(label, val_f1_labels[label]))


if __name__ == '__main__':
    main()
