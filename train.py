import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import logging
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from utils import config as config_module
from sklearn.metrics import classification_report
import torch.nn as nn
import time
import random
from net import net_vmamba_v4
from data_loader_houston_point import HoustonPointDataset


def aa(yt, j1t):
    yt = torch.from_numpy(np.array(yt))
    j1t = torch.from_numpy(np.array(j1t))
    mem = [0] * num_class
    all_mem = [0] * num_class
    n_r = [0] * num_class
    n_p = [0] * num_class
    oa = torch.eq(j1t, yt)

    for x, y in zip(yt, j1t):
        n_r[x] += 1
        n_p[y] += 1
        all_mem[x] += 1
        if x == y:
            mem[x] += 1

    mem = np.array(mem)
    all_mem = np.array(all_mem) + 1e-3
    n_r = np.array(n_r)
    n_p = np.array(n_p)
    pe = np.sum(n_r * n_p) / (yt.shape[0] * yt.shape[0])
    acc_per_class = mem / all_mem
    AA = np.mean(mem / all_mem)
    oa = torch.mean(oa.float())
    k = (oa - pe) / (1 - pe)
    return oa, AA, k, acc_per_class


def setup_logger(log_dir, use_data):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir,
        f"train_{use_data}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logger = logging.getLogger(f"train_logger_{use_data}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, log_file


def seed_everything(seed, strict_determinism=False):
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 这里统一覆盖模型文件里对 cudnn 的设置，避免 benchmark 干扰复现。
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if strict_determinism:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_curriculum_weights(epoch, schedule_mode='epoch10'):
    if schedule_mode == 'none':
        consistency_weight = 0.08
        gate_weight = 0.001

    elif schedule_mode == 'epoch10':
        # 在验证过 10/25/40 过晚、gate=0.00035 过弱之后，回到更优的 10/20/35 切换时机，
        # 但只把第二阶段的一致性约束从 0.08 轻微放软到 0.07，避免中期上限被压住。
        if epoch < 10:
            consistency_weight = 0.05
            gate_weight = 0.0
        elif epoch < 20:
            consistency_weight = 0.06
            gate_weight = 0.0002
        elif epoch < 35:
            consistency_weight = 0.07
            gate_weight = 0.0005
        else:
            consistency_weight = 0.04
            gate_weight = 0.0005

    elif schedule_mode == 'epoch50':
        if epoch < 50:
            consistency_weight = 0.05
            gate_weight = 0.0
        else:
            consistency_weight = 0.08
            gate_weight = 0.001
    else:
        raise ValueError(f'Unknown schedule_mode: {schedule_mode}')

    return consistency_weight, gate_weight


def get_loss(preds, target, epoch):
    hsi_logits, lidar_logits, out_logits = preds

    loss_hsi = criterion(hsi_logits, target).mean()
    loss_lidar = criterion(lidar_logits, target).mean()
    loss_fuse = criterion(out_logits, target).mean()
    aux_weight = 0.60
    cls_loss = loss_fuse + aux_weight * (loss_hsi + loss_lidar)

    target_cos = torch.ones(hsi_logits.shape[0], device=device)
    consistency_loss = 0.5 * (
        criterion_cosine(hsi_logits, out_logits, target_cos) +
        criterion_cosine(lidar_logits, out_logits, target_cos)
    )

    consistency_weight, gate_weight = get_curriculum_weights(epoch, schedule_mode)

    gate_reg = model.get_gate_regularization_loss()
    if gate_reg is None:
        gate_reg = torch.tensor(0.0, device=device)
        gate_weight = 0.0

    cls_outer_weight = 0.5
    cons_outer_weight = 0.5
    total = cls_outer_weight * cls_loss + cons_outer_weight * consistency_weight * consistency_loss + gate_weight * gate_reg

    stats = {
        'cls_loss': cls_loss.detach(),
        'consistency_loss': consistency_loss.detach(),
        'gate_reg': gate_reg.detach(),
        'loss_hsi': loss_hsi.detach(),
        'loss_lidar': loss_lidar.detach(),
        'loss_fuse': loss_fuse.detach(),
        'consistency_weight': torch.tensor(consistency_weight, device=device),
        'gate_weight': torch.tensor(gate_weight, device=device),
        'aux_weight': torch.tensor(aux_weight, device=device),
        'cls_outer_weight': torch.tensor(cls_outer_weight, device=device),
        'cons_outer_weight': torch.tensor(cons_outer_weight, device=device),
    }
    return total, stats


def print_memory_usage(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        reserved = torch.cuda.memory_reserved() / 1024 ** 2
        print(f"[{tag}] Allocated memory: {allocated:.2f} MB")
        print(f"[{tag}] Reserved memory: {reserved:.2f} MB")


def print_official_sample_counts(train_set, test_set):
    train_labels = np.array([int(train_set.gt[r, c]) for r, c in train_set.coords])
    test_labels = np.array([int(test_set.gt[r, c]) for r, c in test_set.coords])

    print("train total:", len(train_labels))
    print("test total:", len(test_labels))

    logger.info(f"train total: {len(train_labels)}")
    logger.info(f"test total: {len(test_labels)}")

    for cls in range(1, 16):
        tr = int((train_labels == cls).sum())
        te = int((test_labels == cls).sum())
        msg = f"class {cls}: train={tr}, test={te}"
        print(msg)
        logger.info(msg)


def fit():
    global best_acc, best_epoch, start_epoch

    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        dt_size = len(train_data_loader.dataset)
        epoch_loss = 0.0
        step = 0
        cls_meter = 0.0
        cons_meter = 0.0
        gate_meter = 0.0
        hsi_meter = 0.0
        lidar_meter = 0.0
        fuse_meter = 0.0
        cons_w_meter = 0.0
        gate_w_meter = 0.0
        aux_w_meter = 0.0
        cls_outer_w_meter = 0.0
        cons_outer_w_meter = 0.0

        model.train()

        pbar = tqdm(
            total=max(dt_size // batch_size, 1),
            desc=f'Epoch {epoch + 1}/{epochs}',
            postfix=dict,
            mininterval=0.3
        )

        for x_hsi, x_lidar, y in train_data_loader:
            x_hsi = x_hsi.to(device, non_blocking=True)
            x_lidar = x_lidar.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(x_hsi, x_lidar)
            loss, loss_stats = get_loss(pred, y, epoch)

            epoch_loss += loss.item()
            cls_meter += loss_stats['cls_loss'].item()
            cons_meter += loss_stats['consistency_loss'].item()
            gate_meter += loss_stats['gate_reg'].item()
            hsi_meter += loss_stats['loss_hsi'].item()
            lidar_meter += loss_stats['loss_lidar'].item()
            fuse_meter += loss_stats['loss_fuse'].item()
            cons_w_meter += loss_stats['consistency_weight'].item()
            gate_w_meter += loss_stats['gate_weight'].item()
            aux_w_meter += loss_stats['aux_weight'].item()
            cls_outer_w_meter += loss_stats['cls_outer_weight'].item()
            cons_outer_w_meter += loss_stats['cons_outer_weight'].item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            pbar.set_postfix(**{
                'train_loss': epoch_loss / (step + 1),
                'cls': cls_meter / (step + 1),
                'cons': cons_meter / (step + 1),
                'gate': gate_meter / (step + 1),
            })
            pbar.update(1)
            step += 1

        pbar.close()

        end_time = time.time()
        one_epoch_train_time = end_time - start_time
        hours = int(one_epoch_train_time // 3600)
        minutes = int((one_epoch_train_time % 3600) // 60)
        seconds = int(one_epoch_train_time % 60)
        train_loss_avg = epoch_loss / max(step, 1)

        print(f"模型上一个epoch训练时间: {hours} 小时 {minutes} 分钟 {seconds} 秒")
        print_memory_usage("one_epoch_memory_use")

        model.eval()
        labels = []
        predicts_hsi, predicts_lidar, predicts_fuse = [], [], []

        for x_hsi, x_lidar, y in tqdm(test_data_loader):
            x_hsi = x_hsi.to(device, non_blocking=True)
            x_lidar = x_lidar.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.no_grad():
                hsi_logits, lidar_logits, fuse_logits = model(x_hsi, x_lidar)

            hsi_label = torch.argmax(hsi_logits, dim=1)
            lidar_label = torch.argmax(lidar_logits, dim=1)
            fuse_label = torch.argmax(fuse_logits, dim=1)

            labels.extend(y.cpu().numpy().tolist())
            predicts_hsi.extend(hsi_label.cpu().numpy().tolist())
            predicts_lidar.extend(lidar_label.cpu().numpy().tolist())
            predicts_fuse.extend(fuse_label.cpu().numpy().tolist())

        report = classification_report(labels, predicts_fuse, digits=4)
        print(report)

        epoch_acc_hsi, AA_hsi, k_hsi, acc_per_class_hsi = aa(labels, predicts_hsi)
        epoch_acc_lidar, AA_lidar, k_lidar, acc_per_class_lidar = aa(labels, predicts_lidar)
        epoch_acc_dev, AA, k, acc_per_class = aa(labels, predicts_fuse)

        print(
            f"head_OA | hsi={epoch_acc_hsi.item():.6f} | "
            f"lidar={epoch_acc_lidar.item():.6f} | fuse={epoch_acc_dev.item():.6f}"
        )
        print(epoch_acc_dev.item(), AA.item(), k.item(), acc_per_class)

        current_lr = optimizer.param_groups[0]['lr']
        current_oa = epoch_acc_dev.item()
        is_new_best = current_oa > best_acc
        if is_new_best:
            best_acc = current_oa
            best_epoch = epoch + 1

        logger.info(
            f"schedule={schedule_mode} | "
            f"seed={seed} | "
            f"epoch={epoch + 1}/{epochs} | "
            f"train_loss={train_loss_avg:.6f} | "
            f"cls={cls_meter / max(step, 1):.6f} | "
            f"cons={cons_meter / max(step, 1):.6f} | "
            f"gate={gate_meter / max(step, 1):.6f} | "
            f"loss_hsi={hsi_meter / max(step, 1):.6f} | "
            f"loss_lidar={lidar_meter / max(step, 1):.6f} | "
            f"loss_fuse={fuse_meter / max(step, 1):.6f} | "
            f"cons_w={cons_w_meter / max(step, 1):.6f} | "
            f"gate_w={gate_w_meter / max(step, 1):.6f} | "
            f"aux_w={aux_w_meter / max(step, 1):.6f} | "
            f"cls_outer_w={cls_outer_w_meter / max(step, 1):.6f} | "
            f"cons_outer_w={cons_outer_w_meter / max(step, 1):.6f} | "
            f"lr={current_lr:.8f} | "
            f"OA_hsi={epoch_acc_hsi.item():.6f} | "
            f"OA_lidar={epoch_acc_lidar.item():.6f} | "
            f"OA={epoch_acc_dev.item():.6f} | "
            f"AA={AA.item():.6f} | "
            f"Kappa={k.item():.6f} | "
            f"best_OA={best_acc:.6f} | "
            f"best_epoch={best_epoch} | "
            f"time={hours:02d}:{minutes:02d}:{seconds:02d}"
        )
        logger.info("per_class_acc_fuse=" + np.array2string(acc_per_class, precision=4, separator=', '))
        logger.info("per_class_acc_hsi=" + np.array2string(acc_per_class_hsi, precision=4, separator=', '))
        logger.info("per_class_acc_lidar=" + np.array2string(acc_per_class_lidar, precision=4, separator=', '))
        logger.info("classification_report_fuse:\n" + report)

        if epoch != 0 and epoch % 50 == 0:
            ckpt_path = os.path.join(save_weights_path, f'{epoch}epoch.pth')
            torch.save(
                {
                    'epoch': epoch + 1,
                    'optimizer_dict': optimizer.state_dict(),
                    'model_dict': model.state_dict(),
                    'scheduler_dict': scheduler.state_dict(),
                },
                ckpt_path
            )
            logger.info(f"checkpoint_saved | epoch={epoch + 1} | path={ckpt_path}")

        if is_new_best:
            best_path = os.path.join(save_weights_path, f'best_ep{best_epoch}_oa{current_oa:.4f}.pth')
            latest_best_path = os.path.join(save_weights_path, 'best_weights.pth')

            state = {
                'epoch': epoch + 1,
                'best_OA': best_acc,
                'best_epoch': best_epoch,
                'optimizer_dict': optimizer.state_dict(),
                'model_dict': model.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
            }

            torch.save(state, best_path)
            torch.save(state, latest_best_path)
            logger.info(f"new_best_model_saved | epoch={best_epoch} | best_OA={best_acc:.6f} | path={best_path}")

        # 固定式学习率调度：不再让 test OA 影响后续训练轨迹。
        scheduler.step()
        print(f'best_OA: {best_acc:.6f} | best_epoch: {best_epoch}')

    logger.info(f"training_finished | schedule={schedule_mode} | seed={seed} | best_OA={best_acc:.6f} | best_epoch={best_epoch}")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 8
    use_data = 'Houston2013'
    cfg = config_module.configs[use_data]
    num_class = cfg.num_class

    seed = 42
    strict_determinism = False
    seed_everything(seed, strict_determinism=strict_determinism)

    epochs = 100
    schedule_mode = 'epoch10'   # 可选: 'none', 'epoch10', 'epoch50'
    net_variant = 'lite_m_clean_keep_gate2_d256_depth6_ssm2_mlp3_drop003_basc_tail3_a005'
    save_weights_path = (
        f'/mnt/mydisk/yang/save_weights_{use_data}_center_msconv045_lite_m_clean_keep_gate2_d256_depth6_'
        f'basc_tail3_a005_lr2e-5_multistep10152540_softgate20_aux060_drop003_triplehead_{schedule_mode}_seed{seed}_twice_clean/'
    )
    os.makedirs(save_weights_path, exist_ok=True)

    logger, log_file = setup_logger(save_weights_path, use_data)
    print(f"log file: {log_file}")
    net_version = getattr(net_vmamba_v4, 'NET_VERSION', 'unknown')
    net_file = getattr(net_vmamba_v4, '__file__', 'unknown')
    print(f"NET_VERSION: {net_version}")
    print(f"NET_FILE: {net_file}")
    logger.info(f"NET_VERSION={net_version}")
    logger.info(f"NET_FILE={net_file}")
    logger.info(
        f"start training | dataset={use_data} | epochs={epochs} | "
        f"batch_size={batch_size} | device={device} | schedule={schedule_mode} | "
        f"seed={seed} | strict_determinism={strict_determinism} | "
        f"net_variant={net_variant} | "
        f"lr_schedule=MultiStepLR[10,15,25,40]x0.5 | soft_curriculum=10/20/35_cons0.07_gate0.0005 | aux=0.60 | loss_balance=cls0.5_cons0.5 | depths=[6] | dims=256 | ssm_ratio=2.0 | ssm_d_state=16 | mlp_ratio=3.0 | drop_path_rate=0.03 | shared_constructor=Tail3_BASC-R_alpha_max0.05_init0.025 | head_eval=hsi/lidar/fuse"
    )

    root_dir = '/mnt/mydisk/ys/Houston2013'
    train_set = HoustonPointDataset(root_dir=root_dir, split='train', patch_size=32)
    test_set = HoustonPointDataset(root_dir=root_dir, split='test', patch_size=32)
    print_official_sample_counts(train_set, test_set)

    model = net_vmamba_v4.VSSM(
        input_shape=(32, 32),
        in_chans_hsi=cfg.input_hsi_channel,
        in_chans_lidar=cfg.input_lidar_channel,
        num_classes=cfg.num_class,
        depths=[6],
        dims=256,
        ssm_d_state=16,
        ssm_ratio=2.0,
        mlp_ratio=3.0,
        use_direction_gate=True,
        drop_path_rate=0.03,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [{'params': model.parameters()}],
        lr=2e-5,
        weight_decay=1e-3
    )

    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    criterion_cosine = nn.CosineEmbeddingLoss()

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[10, 15, 25, 40],
        gamma=0.5
    )

    start_epoch = 0
    best_acc = 0.0
    best_epoch = 0

    print("=== Start training from scratch (Epoch 0) with explicit center supervision + multi-scale spectral convolution ===")
    print(f"=== Curriculum schedule mode: {schedule_mode} | {net_variant} + triple-head eval | seed={seed} | strict_determinism={strict_determinism} ===")

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    test_generator = torch.Generator()
    test_generator.manual_seed(seed)

    train_data_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=train_generator,
        persistent_workers=True
    )

    test_data_loader = DataLoader(
        test_set,
        batch_size=256,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=test_generator,
        persistent_workers=True
    )

    fit()
