import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import datetime
from pathlib import Path
import shutil

import client
import config
import server
from utils.cfg import MergedConfig, merge_class_attrs, save_cfgs


# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='configs/config.json',
                    help='FL configuration file.')
parser.add_argument('-l', '--log', type=str, default='INFO',
                    help='Log messages level.')
args = parser.parse_args()


# 获取RL服务器的超参数
def get_default_cfg():
    from server.PPO.config import AlgoConfig, GeneralConfig
    general_cfg = GeneralConfig()
    # self.algo_name = self.general_cfg.algo_name
    algo_cfg = AlgoConfig()
    cfgs = {'general_cfg': general_cfg, 'algo_cfg': algo_cfg}
    return cfgs


# 创建输出文件夹
def create_dirs(cfg):
    curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # output_dir = f"./output/FMNIST/{curr_time}_{cfg.method}{cfg.mod}"
    output_dir = f"./output/{curr_time}_{cfg.method}{cfg.mod}"
    setattr(cfg, 'output_dir', output_dir)
    config_dir = f"{cfg.output_dir}/config"
    setattr(cfg, 'config_dir', config_dir)
    res_dir = f"{cfg.output_dir}/results"
    setattr(cfg, 'res_dir', res_dir)
    log_dir = f"{cfg.output_dir}/logs"
    setattr(cfg, 'log_dir', log_dir)
    global_model_dir = f"{cfg.output_dir}/global_model"
    setattr(cfg, 'global_model_dir', global_model_dir)
    reports_dir = f"{cfg.output_dir}/reports"
    setattr(cfg, 'reports_dir', reports_dir)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(config_dir).mkdir(parents=True, exist_ok=True)
    Path(res_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(global_model_dir).mkdir(parents=True, exist_ok=True)


def main():
    # 1.获取FL配置参数
    fl_cfg = config.Config(args.config)

    mode = ''  # Train or Test
    method = {
        "basic": 'FedAvg',
        "cluster": 'Cluster',
        "ppo": 'PPO',
        "ppon": 'PPOn'
    }[fl_cfg.server]

    RL_methods = ['PPO', 'PPOn']

    # 2.配置RL服务器参数及部分FL配置参数
    cfg = MergedConfig()  # merge config
    if method in RL_methods:
        cfgs = get_default_cfg()
        cfg = merge_class_attrs(cfg, cfgs['general_cfg'])
        cfg = merge_class_attrs(cfg, cfgs['algo_cfg'])
        mode = f'-{cfg.mode}'
    setattr(cfg, 'method', method)
    setattr(cfg, 'mod', mode)

    # 将配置文件放入output文件夹
    create_dirs(cfg)
    shutil.copy(args.config, cfg.config_dir)
    if method in RL_methods:
        save_cfgs(cfgs, cfg.config_dir)  # save config

    # 3.选择并初始化server
    fl_server = {
        "basic": server.Server(fl_cfg),
        "cluster": server.Cluster(fl_cfg),
        "ppo": server.PPOServer(fl_cfg),
        "ppon": server.PPOnServer(fl_cfg),
    }[fl_cfg.server]

    # 4.运行
    fl_server.boot(cfg)
    fl_server.run(cfg)


if __name__ == "__main__":
    main()
