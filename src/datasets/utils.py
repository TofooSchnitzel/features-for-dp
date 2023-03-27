from torch import stack, std_mean
from tqdm import tqdm


def calc_mean_std(dataset):
    """ 
    Expects the dataset to consist of tensors of the same size
    """
    images = []
    for data in tqdm(dataset, total=len(dataset), leave=False, desc="calculate mean and std"):
        img = data[0]
        images.append(img)
    
    images = stack(images)

    if images.shape[1] in [1, 3]:  # ugly hack
        dims = (0, *range(2, len(images.shape)))
    else:
        dims = (*range(len(images.shape)),)

    std, mean = std_mean(images, dim=dims)

    return mean, std