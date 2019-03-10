class InputExample(object):
    def __init__(self,
                 guid,
                 text_a,
                 text_b=None,
                 label=None):
        """创建一个输入实例
        Args:
            guid: 每个example拥有唯一的id
            text_a: 第一个句子的原始文本，一般对于文本分类来说，只需要text_a
            text_b: 第二个句子的原始文本，在句子对的任务中才有，分类问题中为None
            label: example对应的标签，对于训练集和验证集应非None，测试集为None
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeature(object):
    def __init__(self,
                 text_a_ids,
                 label_id,
                 text_b_ids=None,
                 text_b_lens=None,
                 text_a_lens=None,
                 text_a_char_ids=None,
                 text_b_char_ids=None):
        self.text_a_ids = text_a_ids
        self.text_b_ids = text_b_ids
        self.text_b_lens = text_b_lens
        self.text_a_lens = text_a_lens
        self.label_id = label_id
        self.text_a_char_ids = text_a_char_ids
        self.text_b_char_ids = text_b_char_ids
