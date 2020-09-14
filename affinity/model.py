import copy
import torch
import torch.nn as nn
from affinity.libs.net_utils import NLM, NLM_dot, NLM_woSoft
from torchvision.models import resnet18
from affinity.libs.autoencoder import encoder3, decoder3, encoder_res18, encoder_res50

from affinity.libs.utils import *


def transform(aff, frame1, target_h=None, target_w=None):
    """
    Given aff, copy from frame1 to construct frame2.
    INPUTS:
     - aff: (h*w)*(h*w) affinity matrix
     - frame1: n*c*h*w feature map
    """
    b, c, h, w = frame1.size()
    if target_h is None:
        target_h = h
    if target_w is None:
        target_w = w
    frame1 = frame1.reshape((b, c, -1))
    try:
        frame2 = torch.bmm(frame1, aff)
    except Exception as e:
        print(e)
        print(frame1.size())
        print(aff.size())
    return frame2.reshape((b, c, target_h, target_w))


class normalize(nn.Module):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std=(1.0, 1.0, 1.0)):
        super(normalize, self).__init__()
        self.mean = nn.Parameter(torch.FloatTensor(mean).cuda(), requires_grad=False)
        self.std = nn.Parameter(torch.FloatTensor(std).cuda(), requires_grad=False)

    def forward(self, frames):
        b, c, h, w = frames.size()
        frames = (frames - self.mean.view(1, 3, 1, 1).repeat(b, 1, h, w)) / self.std.view(1, 3, 1, 1).repeat(b, 1, h, w)
        return frames


def create_flat_grid(F_size, GPU=True):
    """
    INPUTS:
     - F_size: feature size
    OUTPUT:
     - return a standard grid coordinate
    """
    b, c, h, w = F_size
    theta = torch.tensor([[1, 0, 0], [0, 1, 0]])
    theta = theta.unsqueeze(0).repeat(b, 1, 1)
    theta = theta.float()

    # grid is a uniform grid with left top (-1,-1) and right bottom (1,1)
    # b * (h*w) * 2
    grid = torch.nn.functional.affine_grid(theta, F_size)
    grid[:, :, :, 0] = (grid[:, :, :, 0] + 1) / 2 * w
    grid[:, :, :, 1] = (grid[:, :, :, 1] + 1) / 2 * h
    # re-implemented by patrick
    # grid = create_grid(F_size, GPU)
    grid_flat = grid.view(b, -1, 2)
    # re-implemented by patrick

    if (GPU):
        grid_flat = grid_flat.cuda()
    return grid_flat


def coords2bbox(coords, patch_size, h_tar, w_tar):
    """
    INPUTS:
     - coords: coordinates of pixels in the next frame
     - patch_size: patch size
     - h_tar: target image height
     - w_tar: target image widthg
    """
    # change to left-top-right-bottom
    b = coords.size(0)
    center = torch.mean(coords, dim=1)  # b * 2
    center_repeat = center.unsqueeze(1).repeat(1, coords.size(1), 1)
    dis_x = torch.sqrt(torch.pow(coords[:, :, 0] - center_repeat[:, :, 0], 2))
    dis_x = torch.mean(dis_x, dim=1).detach()
    dis_y = torch.sqrt(torch.pow(coords[:, :, 1] - center_repeat[:, :, 1], 2))
    dis_y = torch.mean(dis_y, dim=1).detach()
    left = (center[:, 0] - dis_x * 2).view(b, 1)
    left[left < 0] = 0
    right = (center[:, 0] + dis_x * 2).view(b, 1)
    right[right > w_tar] = w_tar
    top = (center[:, 1] - dis_y * 2).view(b, 1)
    top[top < 0] = 0
    bottom = (center[:, 1] + dis_y * 2).view(b, 1)
    bottom[bottom > h_tar] = h_tar
    new_center = torch.cat((left, top, right, bottom), dim=1)
    return new_center


class track_match_comb(nn.Module):
    def __init__(self, pretrained, encoder_dir=None, decoder_dir=None, temp=1, Resnet="r18", color_switch=True,
                 coord_switch=True, model_scale=False):
        super(track_match_comb, self).__init__()

        if Resnet in "r18":
            self.gray_encoder = encoder_res18(pretrained=pretrained, uselayer=4)
        elif Resnet in "r50":
            self.gray_encoder = encoder_res50(pretrained=pretrained, uselayer=4)
        self.rgb_encoder = encoder3(reduce=True)
        self.decoder = decoder3(reduce=True)

        self.rgb_encoder.load_state_dict(torch.load(encoder_dir))
        self.decoder.load_state_dict(torch.load(decoder_dir))
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.rgb_encoder.parameters():
            param.requires_grad = False

        self.nlm = NLM_woSoft()
        self.normalize = normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        self.softmax = nn.Softmax(dim=1)
        self.temp = temp
        self.grid_flat = None
        self.grid_flat_crop = None
        self.color_switch = color_switch
        self.coord_switch = coord_switch
        self.model_scale = model_scale

    def forward(self, img_ref, img_tar, warm_up=True, patch_size=None, test_result=False, strict_orth=True):
        n, c, h_ref, w_ref = img_ref.size()
        n, c, h_tar, w_tar = img_tar.size()
        gray_ref = copy.deepcopy(img_ref[:, 0].view(n, 1, h_ref, w_ref).repeat(1, 3, 1, 1))
        gray_tar = copy.deepcopy(img_tar[:, 0].view(n, 1, h_tar, w_tar).repeat(1, 3, 1, 1))

        gray_ref = (gray_ref + 1) / 2
        gray_tar = (gray_tar + 1) / 2

        gray_ref = self.normalize(gray_ref)
        gray_tar = self.normalize(gray_tar)

        Fgray1 = self.gray_encoder(gray_ref)
        Fgray2 = self.gray_encoder(gray_tar)
        Fcolor1 = self.rgb_encoder(img_ref)

        f_ref_size = Fgray1.size(2)
        if f_ref_size > patch_size[0]:
            center = f_ref_size // 2
            half_patch_size = patch_size[0] // 2
            bbox = torch.tensor(
                [center - half_patch_size, center - half_patch_size, center - half_patch_size + patch_size[0],
                 center - half_patch_size + patch_size[0]], dtype=torch.float).cuda()
            bbox = bbox.repeat(Fgray1.size(0), 1)
            Fgray1 = diff_crop(Fgray1, bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3],
                               patch_size[1], patch_size[0])
            Fcolor1 = diff_crop(Fcolor1, bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3],
                                patch_size[1], patch_size[0])

        output = []

        if warm_up:
            f_tar_size = Fgray2.size(2)
            if f_tar_size > patch_size[0]:
                center = f_ref_size // 2
                half_patch_size = patch_size[0] // 2
                bbox = torch.tensor(
                    [center - half_patch_size, center - half_patch_size, center - half_patch_size + patch_size[0],
                     center - half_patch_size + patch_size[0]], dtype=torch.float).cuda()
                bbox = bbox.repeat(Fgray2.size(0), 1)
                Fgray2 = diff_crop(Fgray2, bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3],
                                   patch_size[1], patch_size[0])
            aff = self.nlm(Fgray1, Fgray2)
            aff_norm = self.softmax(aff)
            Fcolor2_est = transform(aff_norm, Fcolor1)
            color2_est = self.decoder(Fcolor2_est)

            output.append(color2_est)
            output.append(aff)

            if self.color_switch:
                Fcolor2 = self.rgb_encoder(img_tar)
                f_tar_size = Fcolor2.size(2)
                if f_tar_size > patch_size[0]:
                    center = f_ref_size // 2
                    half_patch_size = patch_size[0] // 2
                    bbox = torch.tensor(
                        [center - half_patch_size, center - half_patch_size, center - half_patch_size + patch_size[0],
                         center - half_patch_size + patch_size[0]], dtype=torch.float).cuda()
                    bbox = bbox.repeat(Fcolor2.size(0), 1)
                    Fcolor2 = diff_crop(Fcolor2, bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3],
                                        patch_size[1], patch_size[0])
                # correct aff-softmax
                Fcolor1_est = transform(self.softmax(aff.transpose(1, 2)), Fcolor2)
                color1_est = self.decoder(Fcolor1_est)
                output.append(color1_est)
        else:
            if (self.grid_flat is None):
                self.grid_flat = create_flat_grid(Fgray2.size())
            aff_ref_tar_o = self.nlm(Fgray1, Fgray2)
            aff_ref_tar = torch.nn.functional.softmax(aff_ref_tar_o * self.temp, dim=2)
            coords = torch.bmm(aff_ref_tar, self.grid_flat)
            center = torch.mean(coords, dim=1)  # b * 2
            fixedbbox = center2bbox(center, patch_size, h_tar, w_tar)
            modeled_bbox = coords2bbox(coords, patch_size, h_tar, w_tar)
            if not self.model_scale:
                bbox = fixedbbox
            # new_c = center2bbox(center, patch_size, Fgray2.size(2), Fgray2.size(3))
            # print("center2bbox:", new_c, h_tar, w_tar)
            else:
                bbox = modeled_bbox
                '''limit_h, limit_w = Fgray2.size(2), Fgray2.size(3)
                expand = (coords - center.view(- 1, 1, 2)).abs().mean(dim=1)  # b*2
                left_top = center - expand  # b*2
                right_bottom = center + expand
                left_top[left_top < 0] = 0
                right_bottom[:, 0][right_bottom[:, 0] > limit_w] = limit_w
                right_bottom[:, 1][right_bottom[:, 1] > limit_h] = limit_h
                bbox = torch.cat([left_top, right_bottom], dim=-1)'''
            if test_result:
                return bbox, aff_ref_tar

            Fgray2_crop = diff_crop(Fgray2, bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3],
                                    patch_size[1], patch_size[0])

            # Fgray2_crop = diff_crop(Fgray2, new_c[:, 0], new_c[:, 2], new_c[:, 1], new_c[:, 3], patch_size[1],
            #                        patch_size[0])
            # print("HERE: ", Fgray2.size(), Fgray1.size(), Fgray2_crop.size())

            aff_p = self.nlm(Fgray1, Fgray2_crop)
            aff_norm = self.softmax(aff_p * self.temp)
            Fcolor2_est = transform(aff_norm, Fcolor1)
            color2_est = self.decoder(Fcolor2_est)

            Fcolor2_full = self.rgb_encoder(img_tar)
            Fcolor2_crop = diff_crop(Fcolor2_full, bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3], patch_size[1],
                                     patch_size[0])

            output.append(color2_est)
            output.append(aff_p)
            output.append((bbox * 8, modeled_bbox * 8))
            output.append(coords)

            # color orthorganal
            if self.color_switch:
                # correct aff-softmax
                if strict_orth:
                    aff_mat = aff_norm.transpose(1, 2)
                else:
                    aff_mat = self.softmax(aff_p.permute(0, 2, 1))
                Fcolor1_est = transform(aff_mat, Fcolor2_crop)
                color1_est = self.decoder(Fcolor1_est)
                output.append(color1_est)

            # coord orthorganal
            if self.coord_switch:
                if strict_orth:
                    aff_norm_tran = aff_norm.transpose(1, 2)
                else:
                    aff_norm_tran = self.softmax(aff_p.permute(0, 2, 1) * self.temp)
                if self.grid_flat_crop is None:
                    self.grid_flat_crop = create_flat_grid(Fgray2_crop.size()).permute(0, 2, 1).detach()
                C12 = torch.bmm(self.grid_flat_crop, aff_norm)
                C11 = torch.bmm(C12, aff_norm_tran)
                output.append(self.grid_flat_crop)
                output.append(C11)

        return output


class Model_switchGTfixdot_swCC_Res(nn.Module):
    def __init__(self, encoder_dir=None, decoder_dir=None, fix_dec=True,
                 temp=None, pretrainRes=False, uselayer=3, model='resnet18'):
        '''
        For switchable concenration loss
        Using Resnet18
        '''
        super(Model_switchGTfixdot_swCC_Res, self).__init__()
        if (model == 'resnet18'):
            print('Use ResNet18.')
            self.gray_encoder = encoder_res18(pretrained=pretrainRes, uselayer=uselayer)
        else:
            print('Use ResNet50.')
            self.gray_encoder = encoder_res50(pretrained=pretrainRes, uselayer=uselayer)
        self.rgb_encoder = encoder3(reduce=True)
        self.nlm = NLM_woSoft()
        self.decoder = decoder3(reduce=True)
        self.temp = temp
        self.softmax = nn.Softmax(dim=1)
        self.cos_window = torch.Tensor(np.outer(np.hanning(40), np.hanning(40))).cuda()
        self.normalize = normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

        self.rgb_encoder.load_state_dict(torch.load(encoder_dir))
        self.decoder.load_state_dict(torch.load(decoder_dir))

        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.rgb_encoder.parameters():
            param.requires_grad = False

    def forward(self, gray1, gray2, color1=None, color2=None):
        # move gray scale image to 0-1 so that they match ImageNet pre-training
        gray1 = (gray1 + 1) / 2
        gray2 = (gray2 + 1) / 2

        # normalize to fit resnet
        b = gray1.size(0)

        gray1 = self.normalize(gray1)
        gray2 = self.normalize(gray2)

        Fgray1 = self.gray_encoder(gray1)
        Fgray2 = self.gray_encoder(gray2)

        aff = self.nlm(Fgray1, Fgray2)  # bx4096x4096
        aff_norm = self.softmax(aff * self.temp)

        if (color1 is None):
            return aff_norm, Fgray1, Fgray2

        Fcolor1 = self.rgb_encoder(color1)
        Fcolor2 = self.rgb_encoder(color2)
        Fcolor2_est = transform(aff_norm, Fcolor1)
        pred2 = self.decoder(Fcolor2_est)

        Fcolor1_est = transform(aff_norm.transpose(1, 2), Fcolor2)
        pred1 = self.decoder(Fcolor1_est)

        return pred1, pred2, aff_norm, aff, Fgray1, Fgray2
