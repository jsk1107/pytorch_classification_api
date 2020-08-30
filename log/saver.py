import os
import datetime
import glob
import torch
import shutil
from log import summarise, logger

TODAY = datetime.datetime.today().strftime('%Y%m%d')


class Saver(object):

    def __init__(self, config):
        self.config = config
        self.directory = os.path.join('run', self.config.project_name, config.model)
        self.today_runs = glob.glob(os.path.join(self.directory, f'{TODAY}_*'))
        self.runs = sorted([int(os.path.basename(run).split('_')[1]) for run in self.today_runs])
        run_id = int(self.runs[-1]) + 1 if self.runs else 0

        self.expriment_dir = os.path.join(self.directory, f'{TODAY}_{run_id}')

        if os.path.exists(self.expriment_dir):
            os.makedirs(self.expriment_dir, exist_ok=True)

    def save_checkpoint(self, state, is_best, k, filename='checkpoint.pth.tar'):
        '''save checkpoint'''

        filename = os.path.join(self.expriment_dir, k, filename)
        if not os.path.exists(os.path.join(self.expriment_dir, k)):
            os.makedirs(os.path.join(self.expriment_dir, k))
        print(1)
        torch.save(state, filename)
        print(2)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.expriment_dir, 'best_pred.txt'), 'w', encoding='utf-8') as t:
                t.write(str(best_pred))

            if self.runs:
                previous_pred = [.0]
                for run_id in self.runs:
                    path = os.path.join(self.directory, f'{TODAY}_{run_id}', 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r', encoding='utf-8') as t:
                            acc = float(t.read())
                            previous_pred.append(acc)
                    else:
                        continue
                max_previous = max(previous_pred)
                if best_pred > max_previous:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))



if __name__ =='__main__':
    print(TODAY)