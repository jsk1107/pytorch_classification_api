import argparse
from parse_config import ParseConfig
from trainer.trainer import Trainer


def run():
    parser = argparse.ArgumentParser('Classification API')
    parser.add_argument('--config-file', '-c', type=str, default='./cfg/config.yaml',
                        help='Config File')
    config = ParseConfig(parser).parse_args()

    print(config)

    trainer = Trainer(config)

    print('Start Epoch: {}'.format(config.start_epoch))
    print('Total Epoch: {}'.format(config.epoch))

    for epoch in range(config.start_epoch, config.epoch):
        trainer.train(epoch)
        trainer.validation(epoch)

    if config.tensorboard:
        trainer.writer.close()

if __name__ == '__main__':
    run()

