import torch


def get_mask(mask):
    mask = mask.unsqueeze(1).expand(-1, mask.shape[-1], -1)
    mask = torch.triu(mask)
    return mask


def get_useful_ones(out, label, mask):
    # get mask, mask the padding and down triangle
    mask = mask.reshape(-1)
    tmp_out = out.reshape(-1, out.shape[-1])
    tmp_label = label.reshape(-1)
    # index select, for gpu speed
    indices = mask.nonzero(as_tuple=False).squeeze(-1).long()
    tmp_out = tmp_out.index_select(0, indices)

    tmp_label = tmp_label.index_select(0, indices)
    # print(tmp_out.shape)

    return tmp_out, tmp_label


def get_soft_slot(biaffine_score, masks):
    tmp_mask = masks
    masks = masks.reshape(-1)
    batch_size = biaffine_score.shape[0]
    num_label = biaffine_score.shape[-1]
    tmp_out = biaffine_score.reshape(-1, biaffine_score.shape[-1])

    # index select, for gpu speed
    indices = masks.nonzero(as_tuple=False).squeeze(-1).long()
    tmp_out = tmp_out.index_select(0, indices)

    soft_slot = []
    start_index = 0
    for i in range(batch_size):
        mask = tmp_mask[i]
        length = int(mask.sum().item())
        end_index = start_index + length
        x = torch.mean(tmp_out[start_index:end_index], dim=0).reshape(1, -1)
        soft_slot.append(x)
        start_index = end_index

    return torch.cat(soft_slot, dim=0).to(biaffine_score.device.type)


def get_pred_entity(cate_pred, span_scores, is_flat_ner=True):
    top_span = []
    for i in range(len(cate_pred)):
        for j in range(i, len(cate_pred)):
            if cate_pred[i][j] > 0:
                tmp = (cate_pred[i][j].item(), i, j, span_scores[i][j].item())
                top_span.append(tmp)
    top_span = sorted(top_span, reverse=True, key=lambda x: x[3])
    res_entity = []
    for t, ns, ne, _ in top_span:
        for _, ts, te, in res_entity:
            if ns < ts <= ne < te or ts < ns <= te < ne:
                # for both nested and flat ner no clash is allowed
                break
            if is_flat_ner and (ns <= ts <= te <= ne or ts <= ns <= ne <= te):
                # for flat ner nested mentions are not allowed
                break
        else:
            res_entity.append((t, ns, ne))
    return set(res_entity)



def read_label(path):
    with open(path, 'r', encoding='utf8') as f:
        label_set = f.read().splitlines()
    return label_set


