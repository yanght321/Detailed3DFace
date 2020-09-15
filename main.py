import glob
from bilinear_model import BilinearModel
import cv2
from utils import color_transfer, tensor2im
import numpy as np
import os
from options import Options
from pix2pixHD.models import create_model
import torch
import torchvision.transforms.functional as F
from PIL import Image


def main():
    opt = Options().parse()
    img_names = []
    for name in os.listdir(opt.input):
        if any(name.endswith(extension) for extension in
               ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff']):
            img_names.append(name)

    bilinear_model = BilinearModel(opt.predef_dir)

    if opt.name == 'dpmap_rig':
        opt.input_nc = 6
        pos_maps = np.load(f'{opt.predef_dir}/posmaps.npz')
        pos_maps = pos_maps.f.arr_0
        pos_maps = torch.from_numpy(pos_maps).unsqueeze(0)

    dpmap_model = create_model(opt)

    for img_name in img_names:
        print(f'\nProcessing {img_name}')
        base_name = os.path.splitext(img_name)[0]
        if not os.path.exists(f'{opt.output}/{base_name}'):
            os.mkdir(f'{opt.output}/{base_name}')

        img = cv2.imread(f'{opt.input}/{img_name}')
        proj_params, verts = bilinear_model.fit_image(img)
        texture = bilinear_model.get_texture(img, proj_params, verts)
        bilinear_model.save_obj(f'{opt.output}/{base_name}/{base_name}.obj', verts, f'./{base_name}.jpg')
        cv2.imwrite(f'{opt.output}/{base_name}/{base_name}.jpg', texture)

        texture = cv2.resize(texture[600:2500, 1100:3000], (1024, 1024)).astype(np.uint8)

        mask = (255 - cv2.imread(f'{opt.predef_dir}/front_mask.png')[:, :, 0]).astype(bool)
        new_pixels = color_transfer(texture[mask][:, np.newaxis, :])
        texture[mask] = new_pixels[:, 0, :]
        texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB).astype(np.float32)
        texture = np.transpose(texture, (2, 0, 1))
        texture = torch.tensor(texture) / 255
        texture = F.normalize(texture, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), True)
        texture = torch.unsqueeze(texture, 0)

        print('Generating displacement maps...')
        dpmap_full = np.zeros((4096, 4096), dtype=np.uint16)
        dpmap_full[...] = 32768
        dpmap_full = Image.fromarray(dpmap_full)
        if opt.name == 'dpmap_rig':
            for i in range(20):
                ipt = torch.cat((texture, pos_maps[:, i * 3:i * 3 + 3]), dim=1)
                dpmap = dpmap_model.inference(ipt, torch.tensor(0))
                dpmap = tensor2im(dpmap.detach()[0], size=(1900, 1900))
                dpmap = Image.fromarray(dpmap)
                dpmap_full.paste(dpmap, (1100, 600, 3000, 2500))
                dpmap_full.save(f'{opt.output}/{base_name}/{base_name}_dpmap_{str(i)}.png')
        else:
            dpmap = dpmap_model.inference(texture, torch.tensor(0))
            dpmap = tensor2im(dpmap.detach()[0], size=(1900, 1900))
            dpmap = Image.fromarray(dpmap)
            dpmap_full.paste(dpmap, (1100, 600, 3000, 2500))
            dpmap_full.save(f'{opt.output}/{base_name}/{base_name}_dpmap.png')


if __name__ == '__main__':
    main()
