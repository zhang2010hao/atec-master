import configs.basic_config as args
import os
import csv
from data_processing.inputs import InputExample


class DataProcessor(object):
    """数据预处理的基类，自定义的MyPro继承该类"""

    def get_train_examples(self, data_dir):
        """读取训练集 Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """读取验证集 Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """读取标签 Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """读csv文件"""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MyPro(DataProcessor):
    def _create_example(self, lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            guid = "%s-%d" % (set_type, i)
            label = line[0]
            text_a = line[1]
            text_b = line[2]
            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            examples.append(example)
        return examples

    def get_examples(self, mode, data_dir):
        if mode == 'train':
            lines = self._read_tsv(os.path.join(data_dir, "train.csv"))
            examples = self._create_example(lines, mode)
        else:
            lines = self._read_tsv(os.path.join(data_dir, "valid.csv"))
            examples = self._create_example(lines, mode)
        return examples

    def get_labels(self):
        return args.labels
