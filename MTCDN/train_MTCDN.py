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
from torchvision.utils import save_image, make_grid
from data import get_training_set, get_test_set
import  itertools
from Model.utils import *
from PIL import  Image
if __name__ == '__main__':
# Training settings
    parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
    parser.add_argument('--dataset', required=False, default='Gloucester')
    parser.add_argument('--batch_size', type=int, default=2, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--direction', type=str, default='a2b', help='a2b or b2a')#what is the function of direction?
    parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
    parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
    parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=512, help='discriminator filters in first conv layer')
    parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count')
    parser.add_argument('--niter', type=int, default=500, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--epoch', type=int, default=0, help='# of iter at starting learning rate')
    parser.add_argument('--G_lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--D_lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
    parser.add_argument('--lr_decay_iters', type=int, default=100, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
    parser.add_argument("--img_height", type=int, default=256, help="size of image height")
    parser.add_argument("--img_width", type=int, default=256, help="size of image width")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--sample_interval", type=int, default=5, help="interval between saving generator outputs")
    parser.add_argument("--checkpoint_interval", type=int, default=20, help="interval between saving model checkpoints")
    parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
    parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
    parser.add_argument("--lambda_id", type=float, default=5, help="identity loss weight")
    opt = parser.parse_args()
    print(opt)
    model_name = "GloucesterIrecon5"
    os.makedirs("result/%s_%s" % (opt.dataset,model_name), exist_ok=True)
    os.makedirs("result/%s_%s/images" % (opt.dataset,model_name), exist_ok=True)
    os.makedirs("checkpoint/%s_%s" % (opt.dataset,model_name), exist_ok=True)

    print('===> Loading datasets')
    root_path = "dataset/"
    train_set = get_training_set(root_path + opt.dataset, opt.direction)
    test_set = get_test_set(root_path + opt.dataset, opt.direction)

    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batch_size, shuffle=True,drop_last=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=opt.test_batch_size,shuffle=False,drop_last=True)

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


    print('===> Building models')


    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    CD_criterion = FocalLoss(gamma=2, alpha=0.25)

    input_shape = (opt.input_nc, opt.img_height, opt.img_width)
  # input_shape2 = (opt.input_nc*2, opt.img_height, opt.img_width)

    # Initialize generator and discriminator
    G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
    G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
    D_A = Discriminator(input_shape)
    D_B = Discriminator(input_shape)


    G_AB = G_AB.to(device)
    G_BA = G_BA.to(device)
    D_A = D_A.to(device)
    D_B = D_B.to(device)
    criterion_GAN.to(device)
    criterion_cycle.to(device)
    criterion_identity.to(device)


    if opt.epoch != 0:
        # Load pretrained models
        G_AB.load_state_dict(torch.load("checkpoint/%s_%s/G_AB_best.pth" % (opt.dataset,model_name)))
        G_BA.load_state_dict(torch.load("checkpoint/%s_%s/G_BA_best.pth" % (opt.dataset,model_name)))
        D_A.load_state_dict(torch.load("checkpoint/%s_%s/D_A_best.pth" % (opt.dataset,model_name)))
        D_B.load_state_dict(torch.load("checkpoint/%s_%s/D_B_best.pth" % (opt.dataset,model_name)))
    else:
        # Initialize weights
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)
    K = 0
    G_AB_model_path = ("checkpoint/%s_%s/G_AB_best.pth" % (opt.dataset,model_name))
    G_BA_model_path = ("checkpoint/%s_%s/G_BA_best.pth" % (opt.dataset,model_name))
    D_A_model_path = ("checkpoint/%s_%s/D_A_best.pth" % (opt.dataset,model_name))
    D_B_model_path = ("checkpoint/%s_%s/D_B_best.pth" % (opt.dataset,model_name))
    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.G_lr, betas=(opt.b1, opt.b2)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.D_lr, betas=(opt.b1, opt.b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.D_lr, betas=(opt.b1, opt.b2))

    # Learning rate update schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(opt.niter, opt.epoch, opt.niter_decay).step
    )
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=LambdaLR(opt.niter, opt.epoch, opt.niter_decay).step
    )
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=LambdaLR(opt.niter, opt.epoch, opt.niter_decay).step
    )

    # Buffers of previously generated samples
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()
    real_A_buffer = ReplayBuffer()
    real_B_buffer = ReplayBuffer()
    lbl_buffer = ReplayBuffer()
    # setup optimizer
    def sample_images(real_A,real_B,batches_done):
        """Saves a generated sample from the test set"""
        G_AB.eval()
        G_BA.eval()
        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)
        # Arange images along x-axis
        real_A = make_grid(real_A, nrow=5, normalize=True)
        real_B = make_grid(real_B, nrow=5, normalize=True)
        fake_A = make_grid(fake_A, nrow=5, normalize=True)
        fake_B = make_grid(fake_B, nrow=5, normalize=True)
        # Arange images along y-axis
        image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
        save_image(image_grid, "result/%s_%s/images/%s.png" % (opt.dataset,model_name, batches_done), normalize=False)

    def placeholder_file(path):
        """
        Creates an empty file at the given path if it doesn't already exists
        :param path: relative path of the file to be created
        """
        import os
        if not os.path.exists(path):
            with open(path, 'w'): pass

    no_optim_op,no_optim_sar=0,0
    best_F1 = 0
    no_optim = 0
    # count = 0
    prev_time = time.time()
    for epoch in range(opt.epoch_count, opt.niter+ 1):
        # train
        loss_v = []
        A_TP = A_TN = A_FP = A_FN = 0
        B_TP = B_TN = B_FP = B_FN = 0
        for iteration, batch in enumerate(training_data_loader, 1):
            # forward
            real_A, real_B, lbl = batch[0].to(device, dtype=torch.float), batch[1].to(device, dtype=torch.float), \
                                    batch[2].to(device, dtype=torch.long)

            G_AB.train()
            G_BA.train()
            D_A.train()
            D_B.train()
            optimizer_G.zero_grad()
            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss


            valid = Variable(torch.tensor(np.ones((real_A.shape[0],1,16,16))), requires_grad=False).to(device, dtype=torch.float)
            fake = Variable(torch.tensor(np.zeros((real_A.shape[0],1,16,16))), requires_grad=False).to(device, dtype=torch.float)

            loss_GAN_AB = criterion_GAN(D_B(real_B,fake_B)[1], valid)
            loss_GAN_BA = criterion_GAN(D_A(real_A,fake_A)[1], valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
            ######################
            # (1) Update D network
            ######################

            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity

            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------

            optimizer_D_A.zero_grad()

            # Real loss

            # Fake loss (on batch of previously generated samples)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            real_A_ = real_A_buffer.push_and_pop(real_A)
            real_B_ = real_B_buffer.push_and_pop(real_B)
            lbl_ = lbl_buffer.push_and_pop(lbl)
           # loss_real = criterion_GAN(D_A(fake_A, real_A)[0], valid)
            [real_A_logit,fake_A_logit,outputA] = D_A(real_A_.detach(),fake_A_.detach())
            [real_B_logit,fake_B_logit,outputB] = D_B(real_B_.detach(),fake_B_.detach())
            loss_real = criterion_GAN(real_A_logit,valid)

            CD_loss_B = CD_criterion(outputB, lbl_)
            CD_loss_A = CD_criterion(outputA,lbl_)

            loss_fake = criterion_GAN(fake_A_logit, fake)
            # Total loss
            if epoch >= 40:
                loss_D_A = (loss_real + loss_fake)+CD_loss_A*5
                #loss_D_A = (loss_real + loss_fake) / 2
            elif epoch>=20 and epoch<40:
                loss_D_A = (loss_real + loss_fake) + CD_loss_A
            else:
                loss_D_A = (loss_real + loss_fake) / 2

            loss_D_A.backward()
            optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------

            optimizer_D_B.zero_grad()

            # Real loss
            #loss_real = criterion_GAN(D_B(fake_B,real_B)[0], valid)
            loss_real = criterion_GAN(real_B_logit, valid)
            # Fake loss (on batch of previously generated samples)
            loss_fake = criterion_GAN(fake_B_logit, fake)
            # Total loss

            if epoch>=40:
                loss_D_B = (loss_real + loss_fake) + CD_loss_B*5
            elif epoch>=20 and epoch<40:
                loss_D_B = (loss_real + loss_fake) + CD_loss_B
            else:
                loss_D_B = (loss_real + loss_fake) / 2

            loss_D_B.backward()
            optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2

            # --------------
            #  Log Progress
            # --------------
            batches_done = epoch
            batches_left = opt.niter * len(training_data_loader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            if batches_done % opt.sample_interval == 0:
                sample_images(real_A, real_B, batches_done)
            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                % (
                    epoch,
                    opt.niter,
                    iteration,
                    len(training_data_loader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    loss_identity.item(),
                    time_left,
                )
            )
        if epoch>=0:

            for iteration, batch in enumerate(testing_data_loader, 1):
                # forward

                real_A, real_B, lbl = batch[0].to(device, dtype=torch.float), batch[1].to(device, dtype=torch.float), \
                                      batch[2].to(device, dtype=torch.long)
                # real_A, real_B, lbl = batch[0], batch[1], batch[2]
                G_AB.eval()
                G_BA.eval()
                D_A.eval()
                D_B.eval()

                fake_B = G_AB(real_A)
                fake_A = G_BA(real_B)



                [_, _, outputA] = D_A(real_A.detach(), fake_A.detach())
                [_, _, outputB] = D_B(real_B.detach(), fake_B.detach())


                predict2 = torch.argmax(outputA, 1)
                predict2 = predict2.squeeze()

                A_TP += ((predict2 == 1).long() & (lbl == 1).long()).cpu().sum().numpy() + 0.0001
                # TN    predict 和 label 同时为0
                A_TN += ((predict2 == 0).long() & (lbl == 0).long()).cpu().sum().numpy() + 0.0001
                # FN    predict 0 label 1
                A_FN += ((predict2 == 0).long() & (lbl == 1).long()).cpu().sum().numpy() + 0.0001
                # FP    predict 1 label 0
                A_FP += ((predict2 == 1).long() & (lbl == 0).long()).cpu().sum().numpy() + 0.0001
                predict2 = torch.argmax(outputB, 1)
                predict2 = predict2.squeeze()
                B_TP += ((predict2 == 1).long() & (lbl == 1).long()).cpu().sum().numpy() + 0.0001
                # TN    predict 和 label 同时为0
                B_TN += ((predict2 == 0).long() & (lbl == 0).long()).cpu().sum().numpy() + 0.0001
                # FN    predict 0 label 1
                B_FN += ((predict2 == 0).long() & (lbl == 1).long()).cpu().sum().numpy() + 0.0001
                # FP    predict 1 label 0
                B_FP += ((predict2 == 1).long() & (lbl == 0).long()).cpu().sum().numpy() + 0.0001

            precision1 = A_TP / (A_TP + A_FP)
            recall1 = A_TP / (A_TP + A_FN)
            f11 = 2 * recall1 * precision1 / (recall1 + precision1)
            acc1 = (A_TP + A_TN) / (A_TP + A_TN + A_FP + A_FN)

            print()
            print("optical_precition:", precision1)
            print("optical_recall:", recall1)
            print("optical_F1:", f11)
            print("optical_acc:", acc1)

            precision2 = B_TP / (B_TP + B_FP)
            recall2 = B_TP / (B_TP + B_FN)
            f12 = 2 * recall2 * precision2 / (recall2 + precision2)
            acc2 = (B_TP + B_TN) / (B_TP + B_TN + B_FP + B_FN)


            print("SAR_precition:", precision2)
            print("SAR_recall:", recall2)
            print("SAR_F1:", f12)
            print("SAR_acc:", acc2)
        #f1 = (f11+f12) /2.0
        f1 = f11
        #f1 = f12
        if f1 <= best_F1:
            # if valloss >= val_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            best_F1 = f1
            # best_loss_d = loss_d
            if epoch != 0:
                print('Saving model!')
                torch.save(G_AB.state_dict(), G_AB_model_path)
                torch.save(G_BA.state_dict(), G_BA_model_path)
                torch.save(D_A.state_dict(), D_A_model_path)
                torch.save(D_B.state_dict(), D_B_model_path)
        if no_optim > 50 and epoch>=30:
            print('early stop at %d epoch' % epoch)
            break
        if no_optim >= 5 and epoch>=30:
            print('Loading model')
            if os.path.exists(G_AB_model_path):
                G_AB.load_state_dict(torch.load(G_AB_model_path))
                G_BA.load_state_dict(torch.load(G_BA_model_path))
                D_A.load_state_dict(torch.load(D_A_model_path))
                D_B.load_state_dict(torch.load(D_B_model_path))
            else:
                print('No saved Model! Loading Init Model!')
                G_AB.load_state_dict(torch.load(G_AB_model_path))
                G_BA.load_state_dict(torch.load(G_BA_model_path))
                D_A.load_state_dict(torch.load(D_A_model_path))
                D_B.load_state_dict(torch.load(D_B_model_path))
            no_optim = 0
        if epoch != 0 and opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(G_AB.state_dict(), "checkpoint/%s_%s/G_AB_%d.pth" % (opt.dataset,model_name,epoch))
            torch.save(G_BA.state_dict(), "checkpoint/%s_%s/G_BA_%d.pth" % (opt.dataset,model_name,epoch))
            torch.save(D_A.state_dict(), "checkpoint/%s_%s/D_A_%d.pth" % (opt.dataset,model_name,epoch))
            torch.save(D_B.state_dict(), "checkpoint/%s_%s/D_B_%d.pth" % (opt.dataset,model_name,epoch))
        print(best_F1)

