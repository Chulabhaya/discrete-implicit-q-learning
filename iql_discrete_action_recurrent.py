import argparse
import os
import random
import time
from distutils.util import strtobool
import pickle

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import wandb
from common.models import (
    RecurrentDiscreteActor,
    RecurrentDiscreteCritic,
    RecurrentDiscreteValue,
)
from common.replay_buffer import ReplayBuffer
from common.utils import make_env, set_seed, save


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--exp-group", type=str, default=None,
        help="the group under which this experiment falls")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--wandb-project", type=str, default="iql-discrete-action-recurrent",
        help="wandb project name")
    parser.add_argument("--wandb-dir", type=str, default="./",
        help="the wandb directory")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-P-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=100500,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e5),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--history-length", type=int, default=8,
        help="maximum sequence length to sample, None means whole episodes are sampled")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=3e-4,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--v-lr", type=float, default=3e-4,
        help="the learning rate of the state value network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target networks")

    # IQL specific arguments
    parser.add_argument("--iql-tau", type=float, default=0.7,
        help="Expectile value used for value function regression")
    parser.add_argument("--beta", type=float, default=3,
        help="Inverse temperature value used for policy extraction with advantage weighted regression")

    # Offline training specific arguments
    parser.add_argument("--dataset-path", type=str, default="/home/chulabhaya/phd/research/data/mdp_expert/4-8-23_cartpole_p_v0_sac_expert_policy_60_percent_random_data.pkl",
        help="path to dataset for training")
    parser.add_argument("--num-evals", type=int, default=10,
        help="number of evaluation episodes to generate per evaluation during training")
    parser.add_argument("--eval-freq", type=int, default=1000,
        help="timestep frequency at which to run evaluation")

    # Checkpointing specific arguments
    parser.add_argument("--save", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="checkpoint saving during training")
    parser.add_argument("--save-checkpoint-dir", type=str, default="./trained_models/",
        help="path to directory to save checkpoints in")
    parser.add_argument("--checkpoint-interval", type=int, default=5000,
        help="how often to save checkpoints during training (in timesteps)")
    parser.add_argument("--resume", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to resume training from a checkpoint")
    parser.add_argument("--resume-checkpoint-path", type=str, default="./trained_models/CartPole-v0__cql_sac_discrete_action_offline__1__1679358697__xgtu5aa7/global_step_95000.pth",
        help="path to checkpoint to resume training from")
    parser.add_argument("--run-id", type=str, default=None,
        help="wandb unique run id for resuming")

    args = parser.parse_args()
    # fmt: on
    return args


def eval_policy(
    actor,
    env_name,
    seed,
    seed_offset,
    global_step,
    num_evals,
    data_log,
):
    # Put actor model in evaluation mode
    actor.eval()

    with torch.no_grad():
        # Initialization
        env = make_env(
            env_name,
            seed + seed_offset,
        )
        # Track averages
        avg_episodic_return = 0
        avg_episodic_length = 0
        # Start evaluation
        obs, info = env.reset(seed=seed + seed_offset)
        for _ in range(num_evals):
            in_hidden = None
            terminated, truncated = False, False
            while not (truncated or terminated):
                # Get action
                seq_lengths = torch.LongTensor([1])
                action, _, _, out_hidden = actor.get_actions(
                    torch.tensor(obs, dtype=torch.float32).to(device).view(1, 1, -1),
                    seq_lengths,
                    in_hidden,
                )
                action = action.view(-1).detach().cpu().numpy()[0]
                in_hidden = out_hidden

                # Take step in environment
                next_obs, reward, terminated, truncated, info = env.step(action)

                # Update next obs
                obs = next_obs
            avg_episodic_return += info["episode"]["r"][0]
            avg_episodic_length += info["episode"]["l"][0]
            obs, info = env.reset()
        # Update averages
        avg_episodic_return /= num_evals
        avg_episodic_length /= num_evals
        print(
            f"global_step={global_step}, episodic_return={avg_episodic_return}, episodic_length={avg_episodic_length}",
            flush=True,
        )
        data_log["misc/episodic_return"] = avg_episodic_return
        data_log["misc/episodic_length"] = avg_episodic_length

    # Put actor model back in training mode
    actor.train()


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.exp_name}"
    wandb_id = wandb.util.generate_id()
    run_id = f"{run_name}_{wandb_id}"

    # If a unique wandb run id is given, then resume from that, otherwise
    # generate new run for resuming
    if args.resume and args.run_id is not None:
        wandb.init(
            id=args.run_id,
            dir=args.wandb_dir,
            project=args.wandb_project,
            resume="must",
            mode="offline",
        )
    else:
        wandb.init(
            id=run_id,
            dir=args.wandb_dir,
            project=args.wandb_project,
            config=vars(args),
            name=run_name,
            save_code=True,
            settings=wandb.Settings(code_dir="."),
            mode="offline",
        )

    # Set training device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("Running on the following device: " + device.type, flush=True)

    # Set seeding
    set_seed(args.seed, device)

    # Load checkpoint if resuming
    if args.resume:
        print("Resuming from checkpoint: " + args.resume_checkpoint_path, flush=True)
        checkpoint = torch.load(args.resume_checkpoint_path)

    # Set RNG state for seeds if resuming
    if args.resume:
        random.setstate(checkpoint["rng_states"]["random_rng_state"])
        np.random.set_state(checkpoint["rng_states"]["numpy_rng_state"])
        torch.set_rng_state(checkpoint["rng_states"]["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(checkpoint["rng_states"]["torch_cuda_rng_state"])
            torch.cuda.set_rng_state_all(
                checkpoint["rng_states"]["torch_cuda_rng_state_all"]
            )

    # Env setup
    env = make_env(args.env_id, args.seed)
    assert isinstance(
        env.action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    # Initialize models and optimizers
    actor = RecurrentDiscreteActor(env).to(device)
    vf1 = RecurrentDiscreteValue(env).to(device)
    qf1 = RecurrentDiscreteCritic(env).to(device)
    qf2 = RecurrentDiscreteCritic(env).to(device)
    qf1_target = RecurrentDiscreteCritic(env).to(device)
    qf2_target = RecurrentDiscreteCritic(env).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr
    )
    v_optimizer = optim.Adam(list(vf1.parameters()), lr=args.v_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # If resuming training, load models and optimizers
    if args.resume:
        actor.load_state_dict(checkpoint["model_state_dict"]["actor_state_dict"])
        vf1.load_state_dict(checkpoint["model_state_dict"]["vf1_state_dict"])
        qf1.load_state_dict(checkpoint["model_state_dict"]["qf1_state_dict"])
        qf2.load_state_dict(checkpoint["model_state_dict"]["qf2_state_dict"])
        qf1_target.load_state_dict(
            checkpoint["model_state_dict"]["qf1_target_state_dict"]
        )
        qf2_target.load_state_dict(
            checkpoint["model_state_dict"]["qf2_target_state_dict"]
        )
        q_optimizer.load_state_dict(checkpoint["optimizer_state_dict"]["q_optimizer"])
        actor_optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]["actor_optimizer"]
        )
        v_optimizer.load_state_dict(checkpoint["optimizer_state_dict"]["v_optimizer"])

    # Load dataset
    dataset = pickle.load(open(args.dataset_path, "rb"))

    # Initialize replay buffer
    env.observation_space.dtype = np.float32
    rb = ReplayBuffer(
        size=args.buffer_size,
        episodic=True,
        stateful=False,
        device=device,
    )
    rb.load_buffer(dataset)

    # If resuming training, then load previous replay buffer
    if args.resume:
        rb_data = checkpoint["replay_buffer"]
        rb.load_buffer(rb_data)

    # Start time tracking for run
    start_time = time.time()

    # Start the game
    start_global_step = 0
    # If resuming, update starting step
    if args.resume:
        start_global_step = checkpoint["global_step"] + 1

    in_hidden = None
    obs, info = env.reset(seed=args.seed)
    # Set RNG state for env
    if args.resume:
        env.np_random.bit_generator.state = checkpoint["rng_states"]["env_rng_state"]
        env.action_space.np_random.bit_generator.state = checkpoint["rng_states"][
            "env_action_space_rng_state"
        ]
        env.observation_space.np_random.bit_generator.state = checkpoint["rng_states"][
            "env_obs_space_rng_state"
        ]
    for global_step in range(start_global_step, args.total_timesteps):
        # Store values for data logging for each global step
        data_log = {}

        # sample data from replay buffer
        (
            observations,
            actions,
            next_observations,
            rewards,
            terminateds,
            seq_lengths,
        ) = rb.sample(args.batch_size, args.history_length)

        # ---------- update value ---------- #
        # no grad because target networks are updated separately
        with torch.no_grad():
            # two Q-value estimates for reducing overestimation bias
            qf1_target_values = qf1_target(observations, seq_lengths)
            qf2_target_values = qf2_target(observations, seq_lengths)
            qf1_target_a_values = qf1_target_values.gather(2, actions)
            qf2_target_a_values = qf2_target_values.gather(2, actions)
            min_qf_target_values = torch.min(qf1_target_a_values, qf2_target_a_values)

        # generate mask for value and Q-function losses
        q_loss_mask = torch.unsqueeze(
            torch.arange(torch.max(seq_lengths))[:, None] < seq_lengths[None, :], 2
        ).to(device)
        q_loss_mask_nonzero_elements = torch.sum(q_loss_mask).to(device)

        # calculate value function predictions
        v_pred_values = vf1(observations, seq_lengths)

        # calculate expectile loss (eq. 5 in IQL paper)
        value_diff = min_qf_target_values - v_pred_values
        expectile_mask = (value_diff < 0).float()
        expectile_weight = torch.abs(args.iql_tau - expectile_mask)
        vf_loss = (
            torch.sum((expectile_weight * (value_diff**2)) * q_loss_mask)
            / q_loss_mask_nonzero_elements
        )

        # backprop update
        v_optimizer.zero_grad()
        vf_loss.backward()
        v_optimizer.step()

        # ---------- update critic ---------- #
        # calculate Q-values
        qf1_values = qf1(observations, seq_lengths)
        qf2_values = qf2(observations, seq_lengths)
        qf1_a_values = qf1_values.gather(2, actions)
        qf2_a_values = qf2_values.gather(2, actions)

        # calculate TD target
        v_pred_next_values = vf1(next_observations, seq_lengths).detach()
        next_q_values = rewards + ((1 - terminateds) * args.gamma * v_pred_next_values)

        # calculate Q-function loss (eq. 6 in IQL paper)
        qf1_loss = (
            torch.sum(
                F.mse_loss(qf1_a_values, next_q_values, reduction="none") * q_loss_mask
            )
            / q_loss_mask_nonzero_elements
        )
        qf2_loss = (
            torch.sum(
                F.mse_loss(qf2_a_values, next_q_values, reduction="none") * q_loss_mask
            )
            / q_loss_mask_nonzero_elements
        )

        qf_loss = qf1_loss + qf2_loss

        # backprop update
        q_optimizer.zero_grad()
        qf_loss.backward()
        q_optimizer.step()

        # ---------- update actor ---------- #
        if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
            for _ in range(
                args.policy_frequency
            ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                # calculate advantage
                with torch.no_grad():
                    # calculate Q-values
                    qf1_target_values = qf1_target(observations, seq_lengths)
                    qf2_target_values = qf2_target(observations, seq_lengths)
                    qf1_target_a_values = qf1_target_values.gather(2, actions)
                    qf2_target_a_values = qf2_target_values.gather(2, actions)
                    min_qf_target_values = torch.min(
                        qf1_target_a_values, qf2_target_a_values
                    )

                    # calculate state-values
                    v_pred_values = vf1(observations, seq_lengths)

                # calculate exponentiated advantages and clip
                exp_advantages = torch.exp(
                    args.beta * (min_qf_target_values - v_pred_values)
                )
                exp_advantages = torch.min(
                    exp_advantages,
                    torch.tensor([100.0], dtype=torch.float32, device=device).view(
                        -1, 1
                    ),
                )

                # calculate log action probs for dataset actions
                log_action_probs = actor.evaluate(
                    observations, actions.squeeze(), seq_lengths
                )
                log_action_probs = log_action_probs.view(*log_action_probs.shape, 1)

                # calculate policy loss (eq. 7 in IQL paper)
                actor_loss = (
                    torch.sum((-log_action_probs * exp_advantages) * q_loss_mask)
                    / q_loss_mask_nonzero_elements
                )

                # backprop update
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

        # update the target networks
        if global_step % args.target_network_frequency == 0:
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(
                    args.tau * param.data + (1 - args.tau) * target_param.data
                )  # "update target network weights" line in page 8, algorithm 1,
                # in updated SAC paper
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(
                    args.tau * param.data + (1 - args.tau) * target_param.data
                )

        if global_step % 100 == 0:
            data_log["losses/qf1_values"] = qf1_a_values.mean().item()
            data_log["losses/qf2_values"] = qf2_a_values.mean().item()
            data_log["losses/qf1_loss"] = qf1_loss.item()
            data_log["losses/qf2_loss"] = qf2_loss.item()
            data_log["losses/qf_loss"] = qf_loss.item()
            data_log["losses/vf_loss"] = vf_loss.item()
            data_log["losses/actor_loss"] = actor_loss.item()
            data_log["misc/steps_per_second"] = int(
                global_step / (time.time() - start_time)
            )
            print("SPS:", int(global_step / (time.time() - start_time)), flush=True)

        # Evaluate trained policy
        if (global_step + 1) % args.eval_freq == 0 or global_step == 0:
            eval_policy(
                actor,
                args.env_id,
                args.seed,
                10000,
                global_step,
                args.num_evals,
                data_log,
            )

        data_log["misc/global_step"] = global_step
        wandb.log(data_log, step=global_step)

        # Save checkpoints during training
        if args.save:
            if global_step % args.checkpoint_interval == 0:
                # Save models
                models = {
                    "actor_state_dict": actor.state_dict(),
                    "vf1_state_dict": vf1.state_dict(),
                    "qf1_state_dict": qf1.state_dict(),
                    "qf2_state_dict": qf2.state_dict(),
                    "qf1_target_state_dict": qf1_target.state_dict(),
                    "qf2_target_state_dict": qf2_target.state_dict(),
                }
                # Save optimizers
                optimizers = {
                    "q_optimizer": q_optimizer.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "v_optimizer": v_optimizer.state_dict(),
                }
                # Save replay buffer
                rb_data = rb.save_buffer()
                # Save random states, important for reproducibility
                rng_states = {
                    "random_rng_state": random.getstate(),
                    "numpy_rng_state": np.random.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "env_rng_state": env.np_random.bit_generator.state,
                    "env_action_space_rng_state": env.action_space.np_random.bit_generator.state,
                    "env_obs_space_rng_state": env.observation_space.np_random.bit_generator.state,
                }
                if device.type == "cuda":
                    rng_states["torch_cuda_rng_state"] = torch.cuda.get_rng_state()
                    rng_states[
                        "torch_cuda_rng_state_all"
                    ] = torch.cuda.get_rng_state_all()

                save(
                    run_id,
                    args.save_checkpoint_dir,
                    global_step,
                    models,
                    optimizers,
                    rb_data,
                    rng_states,
                )

    env.close()
