def get_cost(energy_type):
    cost_dict = {"mse_origin": mse_loss, "mse_emb": mse_emb_loss}
    assert energy_type in cost_dict
    return cost_dict[energy_type]


# pylint: disable=unused-argument
def mse_loss(tensor1, tensor2, **kwargs):
    return (tensor1 - tensor2).pow(2).sum() / tensor1.shape[0] / 2
    # if len(squared_diff.shape) > 2:
    #     squared_diff = rearrange(squared_diff, "b c h w -> b (c h w)")
    # return squared_diff.sum(axis=-1).mean() / 2


def mse_emb_loss(img1, img2, **kwargs):
    embedder = kwargs["embedder"]
    squared_diff = (embedder(img1) - embedder(img2)).pow(2)
    assert len(squared_diff.shape) == 2
    return squared_diff.sum(axis=-1).mean()


def label_cost(w_distance_table, source_label, pf_label_probs):
    # w_distance_table: (# source labels, # target labels)
    # source_label: (batch size,) hard labels
    # pf_label_probs: (batch size, # target labels) soft labels
    selected_w_dist = w_distance_table[source_label.long()]
    label_loss = (selected_w_dist * pf_label_probs).sum(axis=1).mean()
    return label_loss
