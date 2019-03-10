# from models.lstm_atten import Model
from models.bimpm import BiMPM
from models.esim import ESIM
from models.esim_2 import ESIM
import time
import torch
import configs.basic_config as config
import os
import torch.optim as optim
from data_processing.pre_processing import prepare_data, create_batch_iter, init_tokenizer, create_features, load_embs
from tqdm import tqdm
from train_valid.train import train
from train_valid.valid import valid


def load_model(path, model):
    if os.path.exists(path + 'model.pkl'):
        model.load_state_dict(torch.load(os.path.join(path, 'model.pkl')))
    return model


def save_model(path, model):
    torch.save(model.state_dict(), os.path.join(path, 'model.pkl'))


def set_all_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_device(n_gpus):
    """
    setup GPU device if available, move model into configured device
    # n_gpus，小于等于0表示使用cpu，大于0则使用range生成list
    # 如果输入的是一个list，则默认使用list[0]作为controller
     """
    if torch.cuda.is_available():
        gpu_list = n_gpus
        list_ids = n_gpus
        if isinstance(n_gpus, int):
            if n_gpus <= 0:
                device = 'cpu'
                list_ids = []

                return device, list_ids
            else:
                gpu_list = range(n_gpus)

        n_gpu = torch.cuda.device_count()
        if len(gpu_list) > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            list_ids = []
        if len(gpu_list) > n_gpu:
            msg = "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
                gpu_list, range(n_gpu))
            print(msg)
            list_ids = range(n_gpu)
        device = torch.device('cuda:%d' % list_ids[0] if len(list_ids) > 0 else 'cpu')
    else:
        device = 'cpu'
        list_ids = []

    return device, list_ids


def set_model_device(model, n_gpus):
    # 设置模型在GPU上还是CPU上

    device, device_ids = prepare_device(n_gpus)
    if len(device_ids) > 1:
        print("current {} GPUs".format(len(device_ids)))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    if len(device_ids) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_ids[0])
    model = model.to(device)
    return model, device, len(device_ids)


# 加载模型
def restore_checkpoint(resume_path, model=None, optimizer=None):
    checkpoint = torch.load(resume_path)
    best = checkpoint['best']
    start_epoch = checkpoint['epoch'] + 1
    if model:
        model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return [model, optimizer, best, start_epoch]


def fit():
    tokenizer = init_tokenizer()
    eval_iter, valid_batch_count = create_batch_iter("dev", tokenizer)
    # epoch_size = num_train_steps * args.train_batch_size * args.gradient_accumulation_steps / args.num_train_epochs

    vocab = tokenizer.vocab
    embs = load_embs(vocab)
    char_vocab = tokenizer.char_vocab
    # model = ESIM(config.hidden_size, config.emb_size, config.linear_size, vocab, embs, dropout=0.1)
    # model = BiMPM(vocab, config.emb_size, config.char_emb_size, config.hidden_size, config.w_size, len(config.labels),
    #               max_word_len=config.max_word_len, char_vocab_size=len(char_vocab),
    #               char_hidden_size=config.char_hidden_size, embs=embs, dropout=0.1)

    model = ESIM(len(vocab),
                 config.emb_size,
                 config.hidden_size,
                 embeddings=embs,
                 padding_idx=0,
                 dropout=0.4,
                 num_classes=2)

    model, device, n_gpu = set_model_device(model, config.gpus)

    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='max',
                                                           factor=0.5,
                                                           patience=0)

    # ------------------------训练------------------------------
    best_f1 = 0.0
    global_step = 0
    for e in range(1, config.num_train_epochs + 1):
        print('----------------------------------------------------------')
        train_iter, train_batch_count = create_batch_iter("train", tokenizer)
        model.train()
        process_bar = tqdm(total=train_batch_count, desc='training')

        loss, acc, f1 = train(model, train_iter, process_bar, optimizer, device, n_gpu, e)
        process_bar.close()

        model.eval()
        process_bar = tqdm(total=valid_batch_count, desc='validding')
        valid_acc, valid_f1 = valid(model, eval_iter, process_bar, device, e)
        scheduler.step(valid_f1)

        print("Epoch: %d finished, valid acc is: %0.6f, valid f1 is: %0.6f" %
              (e,
               valid_acc,
               valid_f1))

        if valid_f1 > best_f1:
            print('this is a best model, saver the new best model')
            best_f1 = valid_f1
            save_model(config.model_path, model)
        else:
            print('this is not a best model')
        process_bar.close()


def main():
    prepare_data(re_gen_data=False)
    fit()


if __name__ == '__main__':
    main()
