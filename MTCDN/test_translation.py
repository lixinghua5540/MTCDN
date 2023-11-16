from __future__ import print_function
import argparse
import os
from math import log10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from loss.focalloss import FocalLoss
from Model.Sun_Net_gan import *
import torchvision.transforms as transforms
from utils import is_image_file, load_img, save_img
from PIL import Image
from Model.utils import *
import time
# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', required=False, default='Gloucester')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=2, help='testing batch size')
parser.add_argument('--direction', type=str, default='a2b', help='a2b or b2a')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=512, help='discriminator filters in first conv layer')
parser.add_argument('--epoch', type=int, default=0, help='# of iter at starting learning rate')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')

parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model_name = "GloucesterIrecon5"
optical_translate_dir = "result/%s_%s/optical_tranlation" % (opt.dataset,model_name)
optical_CD_result_dir = "result/%s_%s/optical_CD_result" % (opt.dataset,model_name)
SAR_translate_dir = "result/%s_%s/SAR_tranlation" % (opt.dataset,model_name)
SAR_CD_result_dir = "result/%s_%s/SAR_CD_result" % (opt.dataset,model_name)
os.makedirs(optical_translate_dir, exist_ok=True)
os.makedirs(optical_CD_result_dir, exist_ok=True)
os.makedirs(SAR_translate_dir, exist_ok=True)
os.makedirs(SAR_CD_result_dir, exist_ok=True)

model_G_AB_path = ("checkpoint/%s_%s/G_AB_best.pth" % (opt.dataset,model_name))
model_G_BA_path = ("checkpoint/%s_%s/G_BA_best.pth" % (opt.dataset,model_name))
model_D_A_path = ("checkpoint/%s_%s/D_A_best.pth" % (opt.dataset,model_name))
model_D_B_path = ("checkpoint/%s_%s/D_B_best.pth" % (opt.dataset,model_name))


# Initialize generator and discriminator


criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
CD_criterion = FocalLoss(gamma=2, alpha=0.25)

input_shape = (opt.input_nc, opt.img_height, opt.img_width)
input_shape2 = (opt.input_nc*2, opt.img_height, opt.img_width)

# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
D_A = Discriminator(input_shape2)
D_B = Discriminator(input_shape2)

cuda = torch.cuda.is_available()

G_AB.load_state_dict(torch.load(model_G_AB_path))
G_BA.load_state_dict(torch.load(model_G_BA_path))
D_A.load_state_dict(torch.load(model_D_A_path))
D_B.load_state_dict(torch.load(model_D_B_path))

G_AB = G_AB.to(device)
G_BA = G_BA.to(device)
D_A = D_A.to(device)
D_B = D_B.to(device)

image_dir = "dataset/{}/test/Image/".format(opt.dataset)#image optical image#a
image_dir2 = "dataset/{}/test/Image2/".format(opt.dataset)#SAR image#b
label_dir = "dataset/{}/test/label/".format(opt.dataset)

image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)
A_TP = A_TN = A_FP = A_FN =0
B_TP = B_TN = B_FP = B_FN =0
G_AB.eval()
G_BA.eval()
D_A.eval()
D_B.eval()
for image_name in image_filenames:
    time1 = time.time()
    img1 = Image.open(image_dir + image_name).convert('RGB')
    img1 = np.array(img1)/255.0
    img1 = np.transpose(img1, (2, 0, 1))
    #
    img2 = Image.open(image_dir2 + image_name).convert('RGB')
    img2 = np.array(img2)/255.0
    img2 = np.transpose(img2, (2, 0, 1))

    label = Image.open(label_dir + image_name)
    label = np.array(label)
    lbl= np.where(label > 0, 1, label)
    lbl = torch.tensor(lbl)

    img1 = torch.tensor(img1)
    img2 = torch.tensor(img2)
    real_A = img1.unsqueeze(0).to(device,dtype=torch.float)
    real_B = img2.unsqueeze(0).to(device, dtype=torch.float)

    lbl = lbl.to(device, dtype=torch.long)

    fake_A = G_BA(real_B)
    fake_B = G_AB(real_A)
    [_,_,outputA] = D_A(real_A,fake_A)#real  op和 SAR转的fake op
    [_,_,outputB] = D_B(real_B,fake_B)#real sar和光学转的 fake SAR

    out_img_A = (fake_A*255).detach().squeeze(0).cpu().numpy().astype(np.uint8)
    out_img_A = np.transpose(out_img_A, (1, 2, 0))
    out_img_B = (fake_B * 255).detach().squeeze(0).cpu().numpy().astype(np.uint8)
    out_img_B = np.transpose(out_img_B, (1, 2, 0))

    predict_A = torch.argmax(outputA, 1)
    predict_A = predict_A.squeeze()


    lbl = lbl.long()
    A_TP += ((predict_A == 1).long() & (lbl == 1).long()).cpu().sum().numpy() + 0.0001
    # TN    predict 和 label 同时为0
    A_TN += ((predict_A == 0).long() & (lbl == 0).long()).cpu().sum().numpy() + 0.0001
    # FN    predict 0 label 1
    A_FN += ((predict_A == 0).long() & (lbl == 1).long()).cpu().sum().numpy() + 0.0001
    # FP    predict 1 label 0
    A_FP += ((predict_A == 1).long() & (lbl == 0).long()).cpu().sum().numpy() + 0.0001
    # print(TP, TN, FN, FP)



    a = predict_A.cpu().detach().numpy().astype(np.uint8)#MTCDN o
    a = np.where(a > 0, 255, a)
    a = Image.fromarray(a)
    a.save(optical_CD_result_dir+'/' + image_name)

    Image.fromarray(out_img_A).save(optical_translate_dir+'/'+image_name)

    predict_B = torch.argmax(outputB, 1)
    predict_B = predict_B.squeeze()

    B_TP += ((predict_B == 1).long() & (lbl == 1).long()).cpu().sum().numpy() + 0.0001
    # TN    predict 和 label 同时为0
    B_TN += ((predict_B== 0).long() & (lbl == 0).long()).cpu().sum().numpy() + 0.0001
    # FN    predict 0 label 1
    B_FN += ((predict_B == 0).long() & (lbl == 1).long()).cpu().sum().numpy() + 0.0001
    # FP    predict 1 label 0
    B_FP += ((predict_B == 1).long() & (lbl == 0).long()).cpu().sum().numpy() + 0.0001
    # print(TP, TN, FN, FP)

    b = predict_B.cpu().detach().numpy().astype(np.uint8)
    b = np.where(b > 0, 255, b)
    b= Image.fromarray(b)
    b.save(SAR_CD_result_dir + '/' + image_name)

    Image.fromarray(out_img_B).save(SAR_translate_dir + '/' + image_name)
    print(time.time()-time1)
 #   save_img(out_img, translate_dir+'/'+image_name)
precision = A_TP / (A_TP + A_FP)
recall = A_TP / (A_TP + A_FN)
f1 = 2 * recall * precision / (recall + precision)
acc = (A_TP + A_TN) / (A_TP + A_TN + A_FP + A_FN)

print("optical_precition:", precision)
print("optical_recall:",recall)
print("optical_F1:", f1)
print("optical_acc:", acc)

precision = B_TP / (B_TP + B_FP)
recall = B_TP / (B_TP + B_FN)
f1 = 2 * recall * precision / (recall + precision)
acc = (B_TP + B_TN) / (B_TP + B_TN + B_FP + B_FN)

print("SAR_precition:", precision)
print("SAR_recall:",recall)
print("SAR_F1:", f1)
print("SAR_acc:", acc)