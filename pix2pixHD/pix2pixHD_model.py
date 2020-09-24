### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
from torch.autograd import Variable
from .base_model import BaseModel
from . import networks


class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        torch.backends.cudnn.benchmark = True
        input_nc = opt.input_nc

        ##### define networks        
        # Generator network
        netG_input_nc = input_nc
        self.netG = networks.define_G(netG_input_nc, 1, 64, 4, 9, 'instance', gpu_ids=self.gpu_ids)

        # load networks
        self.load_network(self.netG, 'G', '')

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None):
        input_label = label_map.data.cuda()
        input_label = Variable(input_label)
        return input_label, inst_map, real_image, feat_map

    def inference(self, label, inst, image=None):
        # Encode Inputs        
        image = Variable(image) if image is not None else None
        input_label, inst_map, real_image, _ = self.encode_input(Variable(label), Variable(inst), image)
        input_concat = input_label

        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(input_concat)
        else:
            fake_image = self.netG.forward(input_concat)
        return fake_image


class InferenceModel(Pix2PixHDModel):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)
