from utils.metrics import loss_fn, evaluate, class_report
import configs.basic_config as config
import torch.nn as nn


def train(model, training_iter, process_bar, optimizer, device, n_gpu, epoch, max_gradient_norm=10.0):
    i = 0
    train_loss = 0.0
    f1 = 0.0
    train_acc = 0.0
    best_throshold = 0.5
    for step, batch in enumerate(training_iter):
        batch = tuple(t.to(device) for t in batch)
        text_a_ids, text_a_lens, text_b_ids, text_b_lens, text_a_char_ids, text_b_char_ids, label_ids = batch

        optimizer.zero_grad()
        logits = model(text_a_ids, text_b_ids, q1_char_inputs=text_a_char_ids, q2_char_inputs=text_b_char_ids,
                       q1_lens=text_a_lens, q2_lens=text_b_lens, device=device)
        train_loss = loss_fn(logits, label_ids)
        if n_gpu > 1:
            train_loss = train_loss.mean()  # mean() to average on multi-gpu.

        train_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()

        logits = logits.cpu()
        label_ids = label_ids.cpu()
        train_acc, f1, best_throshold = evaluate(logits, label_ids, train=False)
        process_bar.set_description(
            "Epoch: %d, Iter_num: %d, Loss: %0.8f, lr: %0.6f, train_acc: %0.6f, f1: %0.6f, best_throshold: %0.6f" %
            (epoch,
             step,
             train_loss.data[0],
             optimizer.param_groups[0]['lr'],
             train_acc,
             f1,
             best_throshold))
        process_bar.update(1)
    return train_loss, train_acc, f1
