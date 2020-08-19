import argparse
from parse_config import ParseConfig
from trainer.trainer import Trainer


def main():
    parser = argparse.ArgumentParser('Dacon Classification')
    parser.add_argument('--config-file', '-c', type=str, default='./Dacon_config.yaml',
                        help='Config File')
    config = ParseConfig(parser).parse_args()

    print(config)
    trainer = Trainer(config)

    print('Start Epoch: {}'.format(config.start_epoch))
    print('Total Epoch: {}'.format(config.epoch))
    for epoch in range(config.start_epoch, config.epoch):
        trainer.train(epoch)
        # if epoch % 5 == 0:
        trainer.validation(epoch)
    trainer.writer.close()

if __name__ == '__main__':
    main()