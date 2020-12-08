import argparse
from parse_config import ParseConfig
from trainer.trainer import Trainer


def main(train=True):
    parser = argparse.ArgumentParser('Classification API')
    parser.add_argument('--config-file', '-c', type=str, default='./config.yaml',
                        help='Config File')
    config = ParseConfig(parser).parse_args()

    print(config)

    if train:
        trainer = Trainer(config)

        print('Start Epoch: {}'.format(config.start_epoch))
        print('Total Epoch: {}'.format(config.epoch))

        for epoch in range(config.start_epoch, config.epoch):
            trainer.train(epoch)
            trainer.validation(epoch)
        trainer.writer.close()
    else:
        pass

if __name__ == '__main__':
    train = True
    main(train)
