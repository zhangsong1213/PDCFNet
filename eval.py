
from __future__ import print_function
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.utils.data import DataLoader
from net.net import net
from data import get_eval_set
# from utils import *
from A import *
import torch.nn.functional as F



parser = argparse.ArgumentParser(description='PyTorch UIE')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--data_test', type=str, default='./Dataset/UIE/UIEB/raw')
parser.add_argument('--label_test', type=str, default='./Dataset/UIE/UIEB/raw')
parser.add_argument('--model', default='weights/epoch_200.pth', help='Pretrained base model')  ## model
parser.add_argument('--output_folder', type=str, default='./results/')

opt = parser.parse_args()


print('===> Loading datasets')
test_set = get_eval_set(opt.data_test, opt.label_test) ## get_eval_set函数中进行了归一化处理0-1
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

print('===> Building model')

model = net().cuda()
model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
print('Pre-trained model is loaded.')


def eval():
    torch.set_grad_enabled(False)
    model.eval()
    print('\nEvaluation:')

    for batch in testing_data_loader:
        with torch.no_grad():
            input, label, name = batch[0], batch[1], batch[2]

            print(name)

        input = input.cuda()

        with torch.no_grad():
            j_out = model(input)

            if not os.path.exists(opt.output_folder):
                os.mkdir(opt.output_folder)
                os.mkdir(opt.output_folder + 'J/')

            j_out_np = np.clip(torch_to_np(j_out), 0, 1)

            my_save_image(name[0], j_out_np, opt.output_folder + 'J/')  ##


if __name__ == '__main__':
    eval()



