from utils.metrics import loss_fn, evaluate, class_report, print_conf_matrix
import configs.basic_config as config
import torch


def valid(model, eval_iter, process_bar, device, epoch):
    # -----------------------验证----------------------------
    count = 0
    y_predicts, y_labels = [], []
    best_throshold = 0.5
    with torch.no_grad():
        for step, batch in enumerate(eval_iter):
            batch = tuple(t.to(device) for t in batch)
            text_a_ids, text_a_lens, text_b_ids, text_b_lens, text_a_char_ids, text_b_char_ids, label_ids = batch
            logits = model(text_a_ids, text_b_ids, q1_char_inputs=text_a_char_ids, q2_char_inputs=text_b_char_ids,
                           q1_lens=text_a_lens, q2_lens=text_b_lens, device=device)
            count += 1
            y_predicts.append(logits)
            y_labels.append(label_ids)

            process_bar.set_description("Epoch: %d, Iter_num: %d" %
                                        (epoch,
                                         step))
            process_bar.update(1)

        eval_predicted = torch.cat(y_predicts, dim=0).cpu()
        eval_labeled = torch.cat(y_labels, dim=0).cpu()

        eval_acc, eval_f1, best_throshold = evaluate(eval_predicted, eval_labeled, train=False)
        print_conf_matrix(eval_predicted, eval_labeled)

        print(
            '\n\nEpoch %d - eval_acc:%6f - eval_f1:%6f - best_throshold:%6f\n'
            % (epoch,
               eval_acc,
               eval_f1,
               best_throshold))

    return eval_acc, eval_f1
