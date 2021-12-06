import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import argparse
import time
from im2scene import config
from im2scene.checkpoints import CheckpointIO
import logging
import telegram_send
import matplotlib.pyplot as plt
import torch
import os
import argparse
from im2scene import config
from im2scene.checkpoints import CheckpointIO
import torch
import os
import argparse
from tqdm import tqdm
import time
from im2scene import config
from im2scene.checkpoints import CheckpointIO
import numpy as np
from im2scene.eval import (
    calculate_activation_statistics, calculate_frechet_distance)
from math import ceil
from torchvision.utils import save_image, make_grid



logger_py = logging.getLogger(__name__)
np.random.seed(0)
torch.manual_seed(0)

# Arguments
parser = argparse.ArgumentParser(
    description='Train a GIRAFFE model.'
)
parser.add_argument('--config', type=str,default='./configs/64res/cars_64.yaml', help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of '
                         'seconds with exit code 2.')
parser.add_argument('--mode', type=str,help='Mode of the file, can be test or demo')

args = parser.parse_args()

cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# Shorthands
out_dir = cfg['training']['out_dir']
backup_every = cfg['training']['backup_every']
exit_after = args.exit_after
lr = cfg['training']['learning_rate']
lr_d = cfg['training']['learning_rate_d']
batch_size = cfg['training']['batch_size']
n_workers = cfg['training']['n_workers']
t0 = time.time()

model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')

# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

train_dataset= config.get_dataset(cfg)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=True,
    pin_memory=True, drop_last=True,
)

model = config.get_model(cfg, device=device, len_dataset=len(train_dataset))


# Initialize training
op = optim.RMSprop if cfg['training']['optimizer'] == 'RMSprop' else optim.Adam
optimizer_kwargs = cfg['training']['optimizer_kwargs']

if hasattr(model, "generator") and model.generator is not None:
    parameters_g = model.generator.parameters()
else:
    parameters_g = list(model.decoder.parameters())
optimizer = op(parameters_g, lr=lr, **optimizer_kwargs)

if hasattr(model, "discriminator") and model.discriminator is not None:
    parameters_d = model.discriminator.parameters()
    optimizer_d = op(parameters_d, lr=lr_d)
else:
    optimizer_d = None

trainer = config.get_trainer(model, optimizer, optimizer_d, cfg, device=device)
checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer,
                             optimizer_d=optimizer_d)

try:
    load_dict = checkpoint_io.load('model.pt')
    print("Loaded model checkpoint.")
except FileExistsError:
    load_dict = dict()
    print("No model checkpoint found.")

epoch_it = load_dict.get('epoch_it', -1)
it = load_dict.get('it', -1)
metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf)

if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf

print('Current best validation metric (%s): %.8f'
      % (model_selection_metric, metric_val_best))

logger = SummaryWriter(os.path.join(out_dir, 'logs'))
# Shorthands
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
validate_every = cfg['training']['validate_every']
visualize_every = cfg['training']['visualize_every']

# Print model
nparameters = sum(p.numel() for p in model.parameters())
#logger_py.info(model)
logger_py.info('Total number of parameters: %d' % nparameters)

if hasattr(model, "discriminator") and model.discriminator is not None:
    nparameters_d = sum(p.numel() for p in model.discriminator.parameters())
    logger_py.info(
        'Total number of discriminator parameters: %d' % nparameters_d)

if hasattr(model, "generator") and model.generator is not None:
    nparameters_g = sum(p.numel() for p in model.generator.parameters())
    logger_py.info('Total number of generator parameters: %d' % nparameters_g)

t0b = time.time()
if args.mode == 'test':
    if not os.path.exists('test'):
        os.makedirs('test')
    image_grid, x = trainer.visualizeTest(it=it)

    out_dir = cfg['training']['out_dir']
    out_dict_file = os.path.join(out_dir, 'fid_evaluation.npz')
    out_img_file = os.path.join(out_dir, 'fid_images.npy')
    out_vis_file = os.path.join(out_dir, 'fid_images.jpg')

    # Model
    model = config.get_model(cfg, device=device)

    checkpoint_io = CheckpointIO(out_dir, model=model)
    checkpoint_io.load(cfg['test']['model_file'])


    # Generate
    model.eval()

    fid_file = cfg['data']['fid_file']
    assert(fid_file is not None)
    fid_dict = np.load(cfg['data']['fid_file'])

    n_images = cfg['test']['n_images']
    batch_size = cfg['training']['batch_size']
    n_iter = ceil(n_images / batch_size)

    out_dict = {'n_images': n_images}

    img_fake = []
    t0 = time.time()
    for i in tqdm(range(n_iter)):
        with torch.no_grad():
            img_fake.append(model(batch_size).cpu())
    img_fake = torch.cat(img_fake, dim=0)[:n_images]
    img_fake.clamp_(0., 1.)
    n_images = img_fake.shape[0]

    t = time.time() - t0
    out_dict['time_full'] = t
    out_dict['time_image'] = t / n_images

    img_uint8 = (img_fake * 255).cpu().numpy().astype(np.uint8)
    np.save(out_img_file[:n_images], img_uint8)

    # use uint for eval to fairly compare
    img_fake = torch.from_numpy(img_uint8).float() / 255.
    mu, sigma = calculate_activation_statistics(img_fake)
    out_dict['m'] = mu
    out_dict['sigma'] = sigma

    # calculate FID score and save it to a dictionary
    fid_score = calculate_frechet_distance(mu, sigma, fid_dict['m'], fid_dict['s'])
    out_dict['fid'] = fid_score
    print("FID Score (%d images): %.6f" % (n_images, fid_score))
    np.savez(out_dict_file, **out_dict)

    # Save a grid of 16x16 images for visualization
    save_image(make_grid(img_fake[:256], nrow=16, pad_value=1.), 'test/FIDimages.png')    

    cfg = config.load_config(args.config, 'configs/default.yaml')
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")
    out_dir = cfg['training']['out_dir']
    render_dir = os.path.join(out_dir, cfg['rendering']['render_dir'])
    if not os.path.exists(render_dir):
        os.makedirs(render_dir)

    # Model
    model = config.get_model(cfg, device=device)

    checkpoint_io = CheckpointIO(out_dir, model=model)
    checkpoint_io.load(cfg['test']['model_file'])

    # Generator
    renderer = config.get_renderer(model, cfg, device=device)

    model.eval()
    out = renderer.render_object_rotation('test')

elif args.mode == 'demo':
    if not os.path.exists('demo'):
        os.makedirs('demo')
    image_grid, x,latents = trainer.visualizeDemo(it=it)
    cfg = config.load_config(args.config, 'configs/default.yaml')
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")
    out_dir = cfg['training']['out_dir']
    render_dir = os.path.join(out_dir, cfg['rendering']['render_dir'])
    if not os.path.exists(render_dir):
        os.makedirs(render_dir)

    # Model
    model = config.get_model(cfg, device=device)

    checkpoint_io = CheckpointIO(out_dir, model=model)
    checkpoint_io.load(cfg['test']['model_file'])

    # Generator
    renderer = config.get_renderer(model, cfg, device=device)

    model.eval()
    out = renderer.render_object_rotationDemo('demo',latent_codes = latents)
else: 
    print('The mode can only take "demo" o "test" as values')

