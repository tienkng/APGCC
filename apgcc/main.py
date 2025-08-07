##########################################################################
## Creator: Cihsiang
## Email: f09921058@ntu.edu.tw
##########################################################################
import os, sys
import shutil
from pathlib import Path
import argparse
import datetime
import random
import time

import torch
from tensorboardX import SummaryWriter
import warnings

warnings.filterwarnings("ignore")

# Custom Modules
from engine import *
from datasets import build_dataset
from models import build_model
import util.misc as utils
from util.logger import setup_logger, AvgerageMeter, EvaluateMeter


# In[0]: Parser
def parse_args():
    from config import cfg, merge_from_file, merge_from_list

    parser = argparse.ArgumentParser("APGCC")
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="",
        help="the path to the training config",
    )
    
    parser.add_argument(
        "-t", "--test", action="store_true", default=False, help="Model test"
    )
    
    parser.add_argument(
        "-e", "--export", action="store_true", default=False, help="Export Model ONNX"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "opts",
        help="overwriting the training configfrom commandline",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    if args.config_file != "":
        cfg = merge_from_file(cfg, args.config_file)
    opts = args.opts or []
    filtered_opts = []
    device = args.device
    i = 0
    while i < len(opts):
        if opts[i] == "DEVICE":
            if i + 1 < len(opts):
                device = opts[i + 1]
                i += 2
            else:
                i += 1
        else:
            filtered_opts.append(opts[i])
            i += 1
    cfg = merge_from_list(cfg, filtered_opts)
    cfg.config_file = args.config_file
    cfg.test = args.test
    cfg.export = args.export
    cfg.device = device  # Lưu device vào cfg
    return cfg

# In[1]: Main
def main():
    # Initial Config and environment
    cfg = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(cfg.GPU_ID)
    seed = cfg.SEED
    if seed != None:
        g = torch.Generator()
        g.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        os.environ["PYTHONHASHSEED"] = str(seed)

    # Function
    if cfg.test:
        test(cfg)
    elif cfg.export:
        export(cfg)
    else:
        train(cfg)

# In[2]: Training Function
def train(cfg):
    # makedir
    output_dir = cfg.OUTPUT_DIR
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    shutil.copy(cfg.config_file, output_dir)  # copy config file

    # logging
    num_gpus = torch.cuda.device_count()
    logger = setup_logger("APGCC", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info("Running with config:\n{}".format(cfg))
    logger.info("##############################################################")
    logger.info("# % TRAIN .....  ")
    logger.info("# Config is {%s}" % (cfg.config_file))
    logger.info("# SEED is {%s}" % (cfg.SEED))
    logger.info("# DATASET is {%s}" % (cfg.DATASETS.DATASET))
    logger.info("# DATA_RT is {%s}" % (cfg.DATASETS.DATA_ROOT))
    logger.info("# MODEL.ENCODER is {%s}" % (cfg.MODEL.ENCODER))
    logger.info("# MODEL.DECODER is {%s}" % (cfg.MODEL.DECODER))
    logger.info("# MODEL.CONFIG is {%s}" % (cfg.MODEL.DECODER_kwargs))
    logger.info("# LOSS.WEIGHT is {%s}" % (cfg.MODEL.WEIGHT_DICT))
    logger.info("# AUXILIARY MODE is {%s}" % (cfg.MODEL.AUX_EN))
    logger.info("# RESUME is {%s}" % (cfg.RESUME))
    logger.info(
        "# BATCH is {%d*%d*%d}" % (cfg.SOLVER.BATCH_SIZE, cfg.MODEL.ROW, cfg.MODEL.LINE)
    )
    logger.info("# OUTPUT_DIR is {%s}" % (cfg.OUTPUT_DIR))
    logger.info("##############################################################")
    logger.info("Eval Log %s" % time.strftime("%c"))

    # Define the dataloader
    train_dl, val_dl = build_dataset(cfg=cfg)
    torch.multiprocessing.set_sharing_strategy(
        "file_system"
    )  # avoid limitation of number of open files.

    # Building the Model & Optimizer
    model, criterion = build_model(cfg=cfg, training=True)
    model.cuda()
    criterion.cuda()

    # Build Trainier
    trainer = Trainer(cfg, model, train_dl, val_dl, criterion)
    print("Start training")
    start_time = time.time()
    for epoch in range(trainer.train_epoch, trainer.epochs):
        for batch in trainer.train_dl:
            trainer.step(batch)
            trainer.handle_new_batch()
        trainer.handle_new_epoch()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time [%s]" % (total_time_str))


# In[3]: Testing Function
def test(cfg):
    # makedir
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    source_dir = cfg.OUTPUT_DIR
    output_dir = os.path.join(
        source_dir, "%s_%.2f" % (cfg.DATASETS.DATASET, cfg.TEST.THRESHOLD)
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    vis_val_path = None
    if cfg.VIS:
        vis_val_path = os.path.join(output_dir, "sample_result_for_val/")
        if not os.path.exists(vis_val_path):
            os.makedirs(vis_val_path)

    # logging
    num_gpus = torch.cuda.device_count()
    logger = setup_logger("AGPCC", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info("Running with config:\n{}".format(cfg))
    logger.info("##############################################################")
    logger.info("# % TEST .....  ")
    logger.info("# Config is {%s}" % (cfg.config_file))
    logger.info("# SEED is {%s}" % (cfg.SEED))
    logger.info("# DATASET is {%s}" % (cfg.DATASETS.DATASET))
    logger.info("# DATA_RT is {%s}" % (cfg.DATASETS.DATA_ROOT))
    logger.info("# MODEL.ENCODER is {%s}" % (cfg.MODEL.ENCODER))
    logger.info("# MODEL.DECODER is {%s}" % (cfg.MODEL.DECODER))
    logger.info("# RESUME is {%s}" % (cfg.RESUME))
    logger.info(
        "# BATCH is {%d*%d*%d}" % (cfg.SOLVER.BATCH_SIZE, cfg.MODEL.ROW, cfg.MODEL.LINE)
    )
    logger.info("# OUTPUT_DIR is {%s}" % (cfg.OUTPUT_DIR))
    logger.info("# RESULT_DIR is {%s}" % (output_dir))
    logger.info("##############################################################")
    logger.info("Eval Log %s" % time.strftime("%c"))

    # Define the dataset.
    train_dl, val_dl = build_dataset(cfg=cfg)
    torch.multiprocessing.set_sharing_strategy(
        "file_system"
    )  # avoid limitation of number of open files.
    num_samples = len(val_dl.dataset)

    # Building the Model & Optimizer
    model = build_model(cfg=cfg, training=False)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("number of params:%d \n" % n_parameters)

    pretrained_dict = torch.load(cfg.TEST.WEIGHT, map_location="cpu")
    model_dict = model.state_dict()
    param_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(param_dict)
    model.load_state_dict(model_dict)

    # Starting Testing
    print("Start testing")
    t1 = time.time()
    result = evaluate_crowd_counting(
        model, val_dl, next(model.parameters()).device, cfg.TEST.THRESHOLD, vis_val_path
    )
    t2 = time.time()
    
    # logger.info('Eval: MAE=%.6f, MSE=%.6f time:%.2fs \n'%(result[0], result[1], t2 - t1))
    logger.info("Inference time: %.2fs" % ((t2 - t1) / num_samples))


def export(cfg):
    device = torch.device("cpu")
    # output_dir = os.path.join(cfg.OUTPUT_DIR, "export")
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    simplify = True
    # logging
    logger = setup_logger("AGPCC", "", 0)
    logger.info("Running with config:\n{}".format(cfg))
    logger.info("##############################################################")
    logger.info("# EXPORT .....  ")
    logger.info("# Config is {%s}" % (cfg.config_file))
    logger.info("# MODEL.ENCODER is {%s}" % (cfg.MODEL.ENCODER))
    logger.info("# MODEL.DECODER is {%s}" % (cfg.MODEL.DECODER))
    logger.info(
        "# BATCH is {%d*%d*%d}" % (cfg.SOLVER.BATCH_SIZE, cfg.MODEL.ROW, cfg.MODEL.LINE)
    )
    logger.info("##############################################################")

    # Building the Model & Optimizer
    model = build_model(cfg=cfg, training=False)
    
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("number of params:%d \n" % n_parameters)

    pretrained_dict = torch.load(cfg.EXPORT.WEIGHT, map_location="cpu")
    model_dict = model.state_dict()
    param_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(param_dict)
    model.load_state_dict(model_dict)
    model.eval()
    
    # dry run
    batch_size = 1
    im = torch.zeros(batch_size, 3, 1024, 1920).to(device)
    for _ in range(2):
        y = model(im)  # dry runs
    # {'pred_logits', 'pred_points', 'offset'}
    
    try:
        import onnx
    except Exception as e:
        print("Pleas `pip install onnx` before import")
        sys.exit(0)
    
    # define input
    dynamic_axes = {'images': {0: 'batch'}}
    
    # define output
    output_names = ['pred_logits', 'pred_points', 'offset']
    output_axes = {
        'pred_logits' : {0 : 'batch'},
        'pred_points' : {0 : 'batch'},
        'offset' : {0 : 'batch'},
    }
    dynamic_axes.update(output_axes)
    
    # set output path
    f = os.path.splitext(cfg.EXPORT.WEIGHT)[0] + ".onnx"
    
    # Starting Testing
    print("Start Export ONNX")
    torch.onnx.export(model,
                      im,
                      f, 
                      verbose=False,
                      export_params=True,       # store the trained parameter weights inside the model file
                      opset_version=16,
                      do_constant_folding=True, # whether to execute constant folding for optimization
                      input_names=['images'],
                      output_names=output_names,
                      dynamic_axes=dynamic_axes)
    
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    if simplify:
        try:
            import onnxsim

            print('\nStarting to simplify ONNX...')
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, 'assert check failed'
        except Exception as e:
            print(f'Simplifier failure: {e}')

        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        onnx.save(model_onnx,f)
        print('ONNX export success, saved as %s' % f)
    
# %%
if __name__ == "__main__":
    main()
