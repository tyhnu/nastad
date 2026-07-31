import os
import sys
import numpy as np
import time

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import random
import argparse
import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler
from mmengine.config import Config, DictAction
from nastad.models import build_detector
from nastad.datasets import build_dataset, build_dataloader

from nastad.cores import train_one_epoch_supernet, eval_one_epoch_supernet_search_IB, eval_one_epoch_supernet, build_optimizer, build_scheduler
from nastad.utils import (
    set_seed,
    update_workdir,
    create_folder,
    save_config,
    setup_logger,
    ModelEma,
    save_checkpoint,
    save_best_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument("--resume_supernet", type=str, required=True, help="path to a trained IB supernet checkpoint")
    parser.add_argument("--not_eval", action="store_true", help="whether not to eval, only do inference")
    parser.add_argument("--disable_deterministic", action="store_true", help="disable deterministic for faster speed")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--select-num', type=int, default=10)

    parser.add_argument('--population-num', type=int, default=100)

    parser.add_argument('--m_prob', type=float, default=0.2)
    parser.add_argument('--s_prob', type=float, default=0.4)
    parser.add_argument('--crossover-num', type=int, default=25)
    parser.add_argument('--mutation-num', type=int, default=25)
    parser.add_argument('--param-limits', type=float, default=50.0)
    parser.add_argument('--min-param-limits', type=float, default=0.0)
    parser.add_argument("--min_size", type=int, default=10, help="minimum sampled stem length")
    parser.add_argument("--max_size", type=int, default=20, help="maximum sampled stem length")
    parser.add_argument("--resume_ea_path", type=str, default=None, help="resume from a checkpoint")
    parser.add_argument("--work_dir", type=str, default=None, help="override config work directory")

    args = parser.parse_args()
    return args


class EvolutionSearcher(object):
    def __init__(self,
                 args,
                 model,
                 model_without_ddp,
                 val_loader,
                 cfg,
                 model_ema,
                 use_amp,
                 output_dir,
                 logger):
        self.model = model
        self.model_without_ddp = model_without_ddp
        self.args = args
        self.cfg=cfg
        self.logger=logger
        self.model_ema=model_ema
        self.use_amp=use_amp
        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.parameters_limits = args.param_limits
        self.min_parameters_limits = args.min_param_limits
        self.val_loader = val_loader
        self.output_dir = output_dir
        self.s_prob = args.s_prob
        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: []}
        self.epoch = 0
        self.checkpoint_path = args.resume_ea_path
        self.candidates = []
        self.top_accuracies = []
        # self.top_mi = []
        # self.cand_params = []

    def save_checkpoint(self):

        info = {}
        info['top_accuracies'] = self.top_accuracies
        info['memory'] = self.memory
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['keep_top_k'] = self.keep_top_k
        info['epoch'] = self.epoch
        checkpoint_path = os.path.join(self.output_dir, "checkpoint-{}.pth.tar".format(self.epoch))
        torch.save(info, checkpoint_path)
        self.logger.info('save checkpoint to {}'.format(checkpoint_path))
        # logger.info("Resume training from: {}".format(args.resume_supernet))

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            return False
        info = torch.load(self.checkpoint_path, map_location="cpu")
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']
        self.logger.info('load checkpoint to {}'.format(self.checkpoint_path))

        # self.logger.info('load checkpoint from', self.checkpoint_path)
        return True

    def is_legal(self, cand):
        assert isinstance(cand, tuple)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            return False
        if hasattr(self.model, "module") and hasattr(self.model.module, "get_sampled_params_numel"):
            try:
                n_parameters = self.model.module.get_sampled_params_numel(cand)
            except TypeError:
                n_parameters = self.model.module.get_sampled_params_numel(choice=cand)
            info['params'] = n_parameters / 10. ** 6

            if info['params'] > self.parameters_limits:
                self.logger.info('parameters limit exceed: %.2fM > %.2fM', info['params'], self.parameters_limits)
                return False

            if info['params'] < self.min_parameters_limits:
                self.logger.info('under minimum parameters limit: %.2fM < %.2fM', info['params'], self.min_parameters_limits)
                return False

        # print("rank:", utils.get_rank(), cand, info['params'])
        # print("sampled model config: {}".format(sampled_config))
        # print("current rank is {}, cand is {}".format(self.args.rank, cand))
        map,mi=eval_one_epoch_supernet_search_IB(
            self.val_loader,
            self.model,
            self.cfg,
            self.logger,
            self.args.rank,
            choice=cand,
            model_ema=self.model_ema,
            use_amp=self.use_amp,
            world_size=self.args.world_size,
            not_eval=self.args.not_eval
        )
        info['mi'] = mi
        info['acc'] = map
        print("map:", map)
        print("mi:", mi)

        # print("cand:", cand)
        # print("map:", map)
        # info['test_acc'] = test_stats['acc1']

        info['visited'] = True
        # print(info)

        return True

    def update_top_k(self, candidates, *, k, key, reverse=True):
        assert k in self.keep_top_k
        self.logger.info('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            # print(cands)
            # print(self.vis_dict)
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            for cand in cands:
                yield cand

    def get_random_cand(self):
        max_stem_length = self.args.max_size
        min_stem_length = self.args.min_size
        stem_length = np.random.randint(min_stem_length, max_stem_length + 1)

        choice_1 = np.random.randint(3, size=stem_length)

        # branch choice (fixed length 5)
        choice_2 = np.random.randint(3, size=5)

        stem_length_array = np.array([stem_length], dtype=np.int64)
        choice = np.concatenate((stem_length_array, choice_1, choice_2)).tolist()

        return tuple(choice)


    def get_random(self, num):
        self.logger.info('random select ........')
        cand_iter = self.stack_random_cand(self.get_random_cand)
        # print(cand_iter)
        while len(self.candidates) < num:
            cand = next(cand_iter)

            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            # print(cand)
            self.logger.info('random {}/{}'.format(len(self.candidates), num))
        self.logger.info('random_num = {}'.format(len(self.candidates)))

    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        self.logger.info('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = list(random.choice(self.keep_top_k[k]))
            stem_length = cand[0]
            cam_length = stem_length + 5
            random_s = random.random()
            if random_s < m_prob:
                # print("rank{}, before mutation is {},".format(self.args.rank,cand))
                for idx in range(1, cam_length + 1):
                    cand[idx]= random.randint(0,2)
                # print("rank{}, after mutation is {}".format(self.args.rank, cand))
            return tuple(cand)
        cand_iter = self.stack_random_cand(random_func)
        # print(cand_iter)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            # print(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            self.logger.info('mutation {}/{}'.format(len(res), mutation_num))

        self.logger.info('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        self.logger.info('crossover ......')
        res = []
        max_iters = 10 * crossover_num

        def random_func():
            max_attempts = max_iters  # maximum number of attempts
            while max_attempts > 0:
                p1 = list(random.choice(self.keep_top_k[k]))
                p2 = list(random.choice(self.keep_top_k[k]))
                if p1[0] == p2[0]:
                    break
                max_attempts -= 1
            if max_attempts <= 0:
                return tuple(p1)

            stem_length = p1[0]
            total_length = 1 + stem_length + 5

            swap_count = 0
            for idx in range(1, total_length):
                if random.random() < self.s_prob:
                    p1[idx] = p2[idx]
                    swap_count += 1

            if swap_count == 0:
                swap_idx = random.randint(1, total_length - 1)
                p1[swap_idx] = p2[swap_idx]

            return tuple(p1)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            self.logger.info('crossover {}/{}'.format(len(res), crossover_num))

        self.logger.info('crossover_num = {}'.format(len(res)))
        return res


    def search(self):
        self.logger.info(
            'population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
                self.population_num, self.select_num, self.mutation_num, self.crossover_num,
                self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))

        # self.load_checkpoint()

        self.get_random(self.population_num)

        while self.epoch < self.max_epochs:
            self.logger.info('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['mi'])

            self.logger.info('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[self.select_num])))
            tmp_accuracy = []
            self.top_accuracies.append(tmp_accuracy)


            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob)
            crossover = self.get_crossover(self.select_num, self.crossover_num)
            self.candidates = mutation + crossover

            self.get_random(self.population_num)

            self.epoch += 1

            self.save_checkpoint()

            # self.update_top_k(
            #     self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['acc'])
            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['mi'])

            for i, cand in enumerate(self.keep_top_k[self.select_num]):
                self.logger.info('No.{} {}, Top-1 val acc = {},'.format(
                    i + 1, cand, self.vis_dict[cand]['mi']))

def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    # DDP init
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank = int(os.environ["RANK"])
    print(f"Distributed init (rank {args.rank}/{args.world_size}, local rank {args.local_rank})")
    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)

    # set random seed, create work_dir, and save config
    set_seed(args.seed, args.disable_deterministic)
    cfg = update_workdir(cfg, args.id, args.world_size)
    if args.rank == 0:
        create_folder(cfg.work_dir)
        save_config(args.config, cfg.work_dir)

    # setup logger
    logger = setup_logger("Train", save_dir=cfg.work_dir, distributed_rank=args.rank)
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    logger.info(f"Config: \n{cfg.pretty_text}")

    # build dataset
    # test_dataset = build_dataset(cfg.dataset.test, default_args=dict(logger=logger))
    # test_loader = build_dataloader(
    #     test_dataset,
    #     rank=args.rank,
    #     world_size=args.world_size,
    #     shuffle=False,
    #     drop_last=False,
    #     **cfg.solver.test,
    # )

    val_dataset = build_dataset(cfg.dataset.val, default_args=dict(logger=logger))
    val_dataset = build_dataloader(
        val_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=False,
        drop_last=False,
        **cfg.solver.test,
    )

    # build model
    model = build_detector(cfg.model)

    # DDP
    use_static_graph = getattr(cfg.solver, "static_graph", False)
    model = model.to(args.local_rank)
    model = DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=False if use_static_graph else True,
        static_graph=use_static_graph,  # default is False, should be true when use activation checkpointing in E2E
    )
    logger.info(f"Using DDP with total {args.world_size} GPUS...")

    # FP16 compression
    use_fp16_compress = getattr(cfg.solver, "fp16_compress", False)
    if use_fp16_compress:
        logger.info("Using FP16 compression ...")
        model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)

    # Model EMA
    use_ema = getattr(cfg.solver, "ema", False)
    if use_ema:
        logger.info("Using Model EMA...")
        model_ema = ModelEma(model)
    else:
        model_ema = None

    # AMP: automatic mixed precision
    use_amp = getattr(cfg.solver, "amp", False)
    if use_amp:
        logger.info("Using Automatic Mixed Precision...")
        scaler = GradScaler()
    else:
        scaler = None

    # build optimizer and scheduler


    # override the max_epoch
    # resume: reset epoch, load checkpoint / best rmse
    if args.resume_supernet != None:
        logger.info("Resume training from: {}".format(args.resume_supernet))
        device = f"cuda:{args.local_rank}"
        checkpoint = torch.load(args.resume_supernet, map_location=device)
        resume_epoch = checkpoint["epoch"]
        logger.info("Resume epoch is {}".format(resume_epoch))
        model.load_state_dict(checkpoint["state_dict"])
        if model_ema != None:
            model_ema.module.load_state_dict(checkpoint["state_dict_ema"])

        del checkpoint  #  save memory if the model is very large such as ViT-g
        torch.cuda.empty_cache()
    else:
        resume_epoch = -1

    logger.info("Training Starts...\n")

    searcher = EvolutionSearcher(
            args=args,
            model=model,
            model_without_ddp=True,
            val_loader=val_dataset,
            model_ema=model_ema,
            use_amp=use_amp,
            cfg=cfg,
            output_dir=cfg.work_dir,
            logger=logger
        )
    searcher.search()

    logger.info("Training Over...\n")
    # print('total searching time = {:.2f} hours'.format(
    #     (time.time() - t) / 3600))

if __name__ == "__main__":
    main()
