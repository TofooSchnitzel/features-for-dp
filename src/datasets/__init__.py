from pathlib import Path

from datasets import pppp, ham10000


def select_dataloader(config):
    data_dir = Path(config.path.raw)/config.dataset.name

    match config.dataset.name:
        case 'pppp':
            dataloader = pppp.DataLoader(data_dir=data_dir)
        case 'ham10000':
            dataloader = ham10000.DataLoader(data_dir=data_dir)
        case other:
            raise NotImplementedError(f'Not implemented for dataset {config.dataset.name}')
    
    return dataloader