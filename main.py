import argparse
from parse_config import ParseConfig
from trainer.trainer import Trainer
from dataloader import get_dataloader

def main():
    parser = argparse.ArgumentParser('Dacon Classification')
    parser.add_argument('--config-file', '-c', type=str, default='./Dacon_config.yaml',
                        help='Config File')
    config = ParseConfig(parser).parse_args()

    print(config)
    # trainer = Trainer(config)

    print('Start Epoch: {}'.format(config.start_epoch))
    print('Total Epoch: {}'.format(config.epoch))
    LETTER_DICT = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10,
                        'L': 11,
                        'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21,
                        'W': 22, 'X': 23, 'Y': 24, 'Z': 25}
    for k in LETTER_DICT.keys():
        trainer = Trainer(config)
        train_loader, _, _  = get_dataloader(config, k)
        for epoch in range(config.start_epoch, config.epoch):
            print(f'letter : {k}')
            trainer.train(epoch, train_loader, k)
        trainer.writer.close()

if __name__ == '__main__':
    main()