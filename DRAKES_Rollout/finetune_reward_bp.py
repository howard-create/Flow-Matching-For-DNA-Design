# direct reward backpropagation (Hydra version)
import flow_gosai
import fm_dna
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import numpy as np
import oracle
from scipy.stats import pearsonr
import torch
import torch.nn.functional as F
import wandb
import os
import datetime
from utils import set_seed
from copy import deepcopy
from dataloader_gosai import get_dataloaders_gosai
from omegaconf import OmegaConf
import math 
from einops import rearrange

def fine_tune(new_model, new_model_y, new_model_y_eval, old_model, cfg, log_path, save_path, train_loader, val_loader=None, eps=1e-5):
    with open(log_path, 'w') as f:
        f.write(str(cfg) + '\n')

    new_model.config.finetuning.truncate_steps = cfg.truncate_steps
    new_model.config.finetuning.gumbel_softmax_temp = cfg.gumbel_temp
    dt = (1 - eps) / cfg.total_num_steps
    new_model.train()
    torch.set_grad_enabled(True)
    optim = torch.optim.Adam(new_model.parameters(), lr=cfg.learning_rate)
    batch_losses = []
    batch_rewards = []
    before_model = deepcopy(new_model)
    batch = next(iter(train_loader))
    seqs = batch["seqs"].cuda()
    B, L = seqs.shape
    steps = 32
    x = None
    iteration = 4

    for epoch_num in range(cfg.num_epochs):
        rewards = []
        rewards_eval = []
        losses = []
        reward_losses = []
        kl_losses = []
        v = []
        tot_grad_norm = 0.0
        new_model.train()
        batch_size_per_gpu = new_model.config.loader.eval_batch_size

        optim.zero_grad()  # Gradient accumulation init
        for _step in range(cfg.num_accum_steps):
            #x = new_model._sample_Dirichlet(
                #seqs,
                #mask_ratio = 1.0).to(new_model.device)
            
            if new_model.config.total_loss:
                for i in range(iteration):
                    x_out, last_x_list, condt_list, move_chance_t_list, copy_flag_list = new_model._sample_finetune_gradient(
                        steps, B, L, x) #, copy_flag_temp=cfg.copy_flag_temp)
                    #x_out, last_x_list, condt_list, move_chance_t_list, copy_flag_list = old_model._sample_finetune_flow(x) #, copy_flag_temp = cfg.copy_flag_temp)
                    
                    x_input = x_out
                    x_out = x_out[:,:,:-1]
                    x_out = torch.transpose(x_out, 1, 2)
                    preds = new_model_y(x_out).squeeze(-1)
                    reward = preds[:, 0]

                    #sample_argmax = torch.argmax(sample, 2)
                    sample  = x_out
                    #sample = F.gumbel_softmax(sample, tau=cfg.gumbel_temp, hard=True)
                    #sample_argmax = 1.0 * F.one_hot(sample, num_classes=4)
                    sample_argmax = sample

                    preds_argmax = new_model_y(sample_argmax).squeeze(-1)
                    reward_argmax = preds_argmax[:, 0]
                    rewards.append(reward_argmax.detach().cpu().numpy())

                    preds_eval = new_model_y_eval(sample_argmax).squeeze(-1)
                    reward_argmax_eval = preds_eval[:, 0]
                    rewards_eval.append(reward_argmax_eval.detach().cpu().numpy())


                    
                    total_kl = []
                    for random_t in range(cfg.total_num_steps):
                        if cfg.truncate_kl and random_t < cfg.total_num_steps - cfg.truncate_steps:
                            continue
                        #move_chance_t 應該是用不到，之後要處理copy_flag(定義每次跳轉間哪些被轉了)(但感覺copy_flag其實不需要)，還有gumbel_max去做可微sample
                        last_x = last_x_list[random_t]
                        condt = condt_list[random_t]
                        #move_chance_t = move_chance_t_list[random_t]
                        #copy_flag = copy_flag_list[random_t]
                        '''
                            p_x0 = new_model(condt, last_x)
                            p_x0_old = old_model(condt, last_x)
                            p_x0 = rearrange(p_x0, 'b c w -> b w c')
                            p_x0_old = rearrange(p_x0_old, 'b c w -> b w c')
                            p_x0 = p_x0[:, :, :-1]
                            p_x0_old = p_x0_old[:, :, :-1]
                            
                            p_x0 = F.softmax(p_x0, dim = -1)
                            p_x0_old = F.softmax(p_x0_old, dim = -1)
                            #t1 = torch.tensor(random_t, device=new_model.device)
                            #t1_weight = new_model.t_weight(p_x0, t1, device = new_model.device)
                            #p_x0 = p_x0*t1_weight
                            #p_x0_old = p_x0_old*t1_weight
                            #log_p_x0 = torch.log(p_x0)
                            #log_p_x0_old = torch.log(p_x0_old)

                            kl_div =  (-p_x0 + p_x0_old + p_x0 * (log_p_x0 - log_p_x0_old)) / condt[0] #*copy_flag 
                            last_x = rearrange(last_x, 'b c w -> b w c')
                            kl_div = (kl_div * last_x[:, :, :-1]).sum((1, 2))
                            total_kl.append(kl_div)
                        '''
                        
                        p_x0 = new_model(last_x, condt)
                        p_x0_old = old_model(last_x, condt)
                        #p_x0 = rearrange(p_x0, 'b c w -> b w c')
                        #p_x0_old = rearrange(p_x0_old, 'b c w -> b w c')
                        p_x0 = p_x0[:, :, :-1]
                        p_x0_old = p_x0_old[:, :, :-1]
                            
                            
                        p_x0 = F.softmax(p_x0, dim = -1)
                        p_x0_old = F.softmax(p_x0_old, dim = -1)
                        log_p_x0 = torch.log(p_x0)
                        log_p_x0_old = torch.log(p_x0_old)
                        kl_div =  (-p_x0 + p_x0_old + p_x0 * (log_p_x0 - log_p_x0_old)) / condt[0] #*copy_flag 
                            
                            #print(kl_div.shape)
                        last_x = F.one_hot(last_x, num_classes=5)  ###
                        #print(kl_div.shape)
                        #print(last_x.shape) 
                        kl_div = (kl_div * last_x[:, :, :-1]).sum((1, 2))
                        total_kl.append(kl_div)
                    

                    #seqs = sample_argmax  # dim=-1 是對 vocab 維度做 argmax，shape [B, L]

                    if i == 0:
                        mask_seqs, _, _, _ = new_model.prompt_completion(seqs, mask_ratio=0.8)
                    elif i == 1:
                        mask_seqs, _, _, _ = new_model.prompt_completion(seqs, mask_ratio=0.5)
                    elif i == 2:
                        mask_seqs, _, _, _ = new_model.prompt_completion(seqs, mask_ratio=0.2)
                    elif i ==3:
                        x= None
                    #x = mask_seqs

                    #ref_v = old_model.forward(x_input, t)
                    #v = new_model.forward(x_input, t)

                    #v = torch.stack(v, dim = 0)
                    #ref_v = torch.stack(ref_v, dim = 0)
                    #kl_loss = torch.nn.functional.mse_loss(v, ref_v, reduction='mean')


                    current_alpha = ((epoch_num + 1) / cfg.alpha_schedule_warmup * cfg.alpha) if epoch_num < cfg.alpha_schedule_warmup else cfg.alpha
                    kl_loss = torch.stack(total_kl, 1).sum(1).mean()
                    reward_loss = -torch.mean(reward)
                    loss = ((reward_loss) + kl_loss * current_alpha) / (iteration * cfg.num_accum_steps)  # Normalized for gradient accumulation

                    loss.backward()  # accumulate gradient  #這裡怪怪的
                    losses.append(loss.item() * iteration * cfg.num_accum_steps)  # denormalize for logging
                    reward_losses.append(reward_loss.item())
                    #kl_losses.append(kl_loss.item())
                    batch_rewards.append(reward.mean().item())

            # Optimizer update after accumulation
                norm = torch.nn.utils.clip_grad_norm_(new_model.parameters(), cfg.gradnorm_clip)
                tot_grad_norm += norm
                optim.step()
                optim.zero_grad()

                batch_losses.append(np.sum(losses))

            else:
                for i in range(3):
                    sample, last_x_list, condt_list, move_chance_t_list, copy_flag_list = new_model._sample_finetune_gradient(
                        x, eval_sp_size=cfg.batch_size, copy_flag_temp=cfg.copy_flag_temp)

                    sample2 = torch.transpose(sample, 1, 2)
                    preds = new_model_y(sample2).squeeze(-1)
                    reward = preds[:, 0]

                    sample_argmax = torch.argmax(sample, 2)
                    sample_argmax = 1.0 * F.one_hot(sample_argmax, num_classes=4)
                    sample_argmax = torch.transpose(sample_argmax, 1, 2)

                    preds_argmax = new_model_y(sample_argmax).squeeze(-1)
                    reward_argmax = preds_argmax[:, 0]
                    rewards.append(reward_argmax.detach().cpu().numpy())

                    preds_eval = new_model_y_eval(sample_argmax).squeeze(-1)
                    reward_argmax_eval = preds_eval[:, 0]
                    rewards_eval.append(reward_argmax_eval.detach().cpu().numpy())

                    total_kl = []
                    for random_t in range(cfg.total_num_steps):
                        if cfg.truncate_kl and random_t < cfg.total_num_steps - cfg.truncate_steps:
                            continue
                        #move_chance_t 應該是用不到，之後應該要處理copy_flag
                        last_x = last_x_list[random_t]
                        condt = condt_list[random_t]
                        move_chance_t = move_chance_t_list[random_t]
                        copy_flag = copy_flag_list[random_t]

                        log_p_x0 = new_model.forward(last_x, condt)[:, :, :-1]
                        log_p_x0_old = old_model.forward(last_x, condt)[:, :, :-1]
                        p_x0 = log_p_x0.exp()
                        p_x0_old = log_p_x0_old.exp()
                        kl_div = copy_flag * (-p_x0 + p_x0_old + p_x0 * (log_p_x0 - log_p_x0_old)) / condt[0, 0, 0]
                        kl_div = (kl_div * last_x[:, :, :-1]).sum((1, 2))
                        total_kl.append(kl_div)

                    seqs = torch.argmax(log_p_x0, dim=-1)  # dim=-1 是對 vocab 維度做 argmax，shape [B, L]

                    if i == 0:
                        mask_seqs, _, _, _ = new_model.prompt_completion(seqs, mask_ratio=0.8)
                    elif i == 1:
                        mask_seqs, _, _, _ = new_model.prompt_completion(seqs, mask_ratio=0.5)
                    elif i == 2:
                        mask_seqs, _, _, _ = new_model.prompt_completion(seqs, mask_ratio=0.2)
                    x = mask_seqs
                current_alpha = ((epoch_num + 1) / cfg.alpha_schedule_warmup * cfg.alpha) if epoch_num < cfg.alpha_schedule_warmup else cfg.alpha
                intersection = (copy_flag * last_x[:, :, :-1]).sum().item()
                print(f"[DEBUG] intersection sum = {intersection:.2f}")

                kl_loss = torch.stack(total_kl, 1).sum(1).mean()
                reward_loss = -torch.mean(reward)
                loss = (reward_loss + kl_loss * current_alpha) / cfg.num_accum_steps

                loss.backward()
                if (_step + 1) % cfg.num_accum_steps == 0:
                    norm = torch.nn.utils.clip_grad_norm_(new_model.parameters(), cfg.gradnorm_clip)
                    tot_grad_norm += norm
                    optim.step()
                    optim.zero_grad()

                batch_losses.append(loss.cpu().detach().numpy())
                batch_rewards.append(torch.mean(reward).cpu().detach().numpy())
                losses.append(loss.cpu().detach().numpy() * cfg.num_accum_steps)
                reward_losses.append(reward_loss.cpu().detach().numpy())
                kl_losses.append(kl_loss.cpu().detach().numpy())
        print(f"Epoch {epoch_num} Mean reward {np.mean(rewards):.6f} Mean reward eval {np.mean(rewards_eval):.6f} "
                f"Mean grad norm {tot_grad_norm:.6f} Mean loss {np.mean(losses):.6f} "
                f"Mean reward loss {np.mean(reward_losses):.6f} Mean kl loss {np.mean(kl_losses):.6f}")

        if cfg.name != 'debug':
            wandb.log({
                "epoch": epoch_num,
                "mean_reward": np.mean(rewards),
                "mean_reward_eval": np.mean(rewards_eval),
                "mean_grad_norm": tot_grad_norm,
                "mean_loss": np.mean(losses),
                "mean reward loss": np.mean(reward_losses),
                "mean kl loss": np.mean(kl_losses)
            })

        if cfg.validation.enabled and (epoch_num + 1) % cfg.validation.every_n_epochs == 0:
            val_stats = run_validation(new_model, old_model, new_model_y_eval, val_loader or train_loader, cfg, before_model=before_model)
            print(f"[VALID] Epoch {epoch_num+1}: reward {val_stats['val_reward']:.4f}") #, KL {val_stats['val_kl']:.4f}")
            if cfg.validation.log_to_wandb and cfg.name != 'debug':
                wandb.log({
                    "val/reward": val_stats["val_reward"],
                    #"val/kl": val_stats["val_kl"]
                }, step=epoch_num + 1)

        if (epoch_num + 1) % cfg.save_every_n_epochs == 0:
            model_path = os.path.join(save_path, f'model_{epoch_num}.ckpt')
            torch.save(new_model.state_dict(), model_path)
            print(f"Model saved at epoch {epoch_num}")

    if cfg.name != 'debug':
        wandb.finish()

    return batch_losses


@torch.no_grad()
def run_validation(model, old_model, reward_model_eval, dataloader, cfg, before_model):
    model.eval()
    old_model.eval()
    reward_model_eval.eval()

    val_rewards = []
    val_kls = []
    n = 0
    for i, batch in enumerate(dataloader):
        seqs = batch["seqs"].cuda()
        B, L = seqs.shape
        #sample = model._sample_Dirichlet(seqs, mask_ratio=1.0)
        #_, _, completion_bool, _ = model.prompt_completion(seqs=seqs, mask_ratio=0.5)

        #_, log_p_x0 = model.get_per_logps(seqs, mask_seqs_n, completion_bool, backbone="default", new_model=model, old_model=old_model, before_model=before_model)
        #_, log_p_x0_old = old_model.get_per_logps(seqs, mask_seqs_n, completion_bool, backbone="ref_backbone", new_model=model, old_model=old_model, before_model=before_model)

        #p_x0 = log_p_x0.exp()
        #p_x0_old = log_p_x0_old.exp()

        #kl = p_x0 * (log_p_x0 - log_p_x0_old)
        #kl_loss = kl.sum(-1)[completion_bool].mean()
        #val_kls.append(kl_loss.item())
        #generator  = model.generator = old_model.sample(
        #num_sampling_steps = 8,
        #num_samples =  B,
        #sequence_length = L,
        #num_samples=1,
        #sequence_length=L,
        #num_sampling_steps = step,
        #stochasticity=5.0,
        #yield_intermediate=True,
    #)
    #samples = list(generator)[-1]
        #for x_t in generator:
           # x_out = x_t  #, copy_flag_temp=cfg.copy_flag_temp)
        #tokens = torch.argmax(x_out, dim=-1)
        #print(x_out)
        #print(seqs.shape)
        seqs = model._sample(num_steps = 128)
        one_hot = F.one_hot(seqs, num_classes=4).float() #####
        #one_hot = one_hot.unsqueeze(0)
        x = one_hot.permute(0, 2, 1)
        reward = reward_model_eval(x).squeeze(-1).mean()
        val_rewards.append(reward.item())
        n += 1

        if cfg.validation._steps == n:
            break

    return {
        "val_reward": sum(val_rewards) / len(val_rewards),
        #"val_kl": sum(val_kls) / len(val_kls)
    }

if __name__ == "__main__":
    base_path = "/home/ubuntu"
    ckpt_path = os.path.join(base_path, 'outputs_gosai/2025.12.06/164243/checkpoints/best.ckpt')
    log_base_dir = os.path.join(base_path, 'mdlm/reward_bp_results_final')
    GlobalHydra.instance().clear()

    initialize(config_path="configs_gosai", job_name="reward_finetune")
    cfg = compose(config_name="config_gosai.yaml")
    cfg.eval.checkpoint_path = ckpt_path

    curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if cfg.name == 'debug':
        save_path = os.path.join(log_base_dir, cfg.name)
        os.makedirs(save_path, exist_ok=True)
        log_path = os.path.join(save_path, 'log.txt')
    else:
        run_name = f'alpha{cfg.alpha}_accum{cfg.num_accum_steps}_bsz{cfg.batch_size}_truncate{cfg.truncate_steps}_temp{cfg.gumbel_temp}_clip{cfg.gradnorm_clip}_{cfg.name}_{curr_time}'
        save_path = os.path.join(log_base_dir, run_name)
        os.makedirs(save_path, exist_ok=True)
        wandb.init(
            project='reward_bp_final',
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=False),
            dir=save_path
        )
        log_path = os.path.join(save_path, 'log.txt')

    set_seed(cfg.seed, use_cuda=True)

    new_model = fm_dna.DiscreteFlowMatchingNet.load_from_checkpoint(cfg.eval.checkpoint_path, config=cfg)
    old_model = fm_dna.DiscreteFlowMatchingNet.load_from_checkpoint(cfg.eval.checkpoint_path, config=cfg)
    reward_model = oracle.get_gosai_oracle(mode='train').to(new_model.device)
    reward_model_eval = oracle.get_gosai_oracle(mode='eval').to(new_model.device)
    reward_model.eval()
    reward_model_eval.eval()

    train_loader, val_loader, _ = get_dataloaders_gosai(cfg)

    fine_tune(new_model, reward_model, reward_model_eval, old_model, cfg, log_path, save_path, train_loader, val_loader)
