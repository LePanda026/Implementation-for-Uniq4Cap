import torch


def apply_to_sample(f, sample, local_rank):
    ## add check for datasets that return none samples for missing items
    if sample == None or len(sample) == 0:
        return {}

    def _apply(x, local_rank):
        if torch.is_tensor(x):
            return f(x,local_rank)
        elif isinstance(x, dict):
            return {key: _apply(value, local_rank) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x, local_rank) for x in x]
        else:
            return x

    return _apply(sample,local_rank)

def move_to_cuda(sample, local_rank):
    def _move_to_cuda(tensor,local_rank):
        return tensor.cuda(local_rank)

    return apply_to_sample(_move_to_cuda, sample, local_rank)

def prepare_sample(samples, local_rank):
    samples = move_to_cuda(samples, local_rank)
    return samples

