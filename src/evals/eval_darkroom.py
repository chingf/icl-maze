import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt

from src.agents.agent import (
    TransformerAgent,
    OptPolicy,
)
from src.envs.darkroom import (
    DarkroomEnv,
    DarkroomEnvVec,
)
from src.utils import convert_to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EvalDarkroom:
    def __init__(self):
        pass

    def create_env(self, config, goal, i_eval):
        dim = config['dim']
        horizon = config['horizon']
        return DarkroomEnv(dim, goal, horizon)

    def create_vec_env(self, envs):
        return DarkroomEnvVec(envs)

    def online(self, eval_trajs, model, config):
        def _deploy_online_vec(vec_env, controller, n_eps, n_eps_in_context, horizon):
            num_envs = vec_env.num_envs
            context_states = torch.zeros(
                (num_envs, n_eps_in_context, horizon, vec_env.state_dim)).float().to(device)
            context_actions = torch.zeros(
                (num_envs, n_eps_in_context, horizon, vec_env.action_dim)).float().to(device)
            context_next_states = torch.zeros(
                (num_envs, n_eps_in_context, horizon, vec_env.state_dim)).float().to(device)
            context_rewards = torch.zeros(
                (num_envs, n_eps_in_context, horizon, 1)).float().to(device)

            cum_means = []

            # Fill context buffer first
            for i in range(n_eps_in_context):
                print(f"Online episode: {i}")
                batch = {
                    'context_states': context_states[:, :i, :, :].reshape(num_envs, -1, vec_env.state_dim),
                    'context_actions': context_actions[:, :i, :].reshape(num_envs, -1, vec_env.action_dim),
                    'context_next_states': context_next_states[:, :i, :, :].reshape(num_envs, -1, vec_env.state_dim),
                    'context_rewards': context_rewards[:, :i, :, :].reshape(num_envs, -1, 1),
                }
                controller.set_batch(batch)
                states_lnr, actions_lnr, next_states_lnr, rewards_lnr = vec_env.deploy_eval(
                    controller)
                context_states[:, i, :, :] = convert_to_tensor(states_lnr)
                context_actions[:, i, :, :] = convert_to_tensor(actions_lnr)
                context_next_states[:, i, :, :] = convert_to_tensor(next_states_lnr)
                context_rewards[:, i, :, :] = convert_to_tensor(rewards_lnr[:, :, None])

                cum_means.append(np.sum(rewards_lnr, axis=-1))

            # Then roll in new data into context buffer
            for i in range(n_eps_in_context, n_eps):
                print(f"Online episode: {i}")
                batch = { 
                    'context_states': context_states.reshape(num_envs, -1, vec_env.state_dim),
                    'context_actions': context_actions.reshape(num_envs, -1, vec_env.action_dim),
                    'context_next_states': context_next_states.reshape(num_envs, -1, vec_env.state_dim),
                    'context_rewards': context_rewards.reshape(num_envs, -1, 1),
                }
                controller.set_batch(batch)
                states_lnr, actions_lnr, next_states_lnr, rewards_lnr = vec_env.deploy_eval(
                    controller)  # Deploy controller in environment for HORIZON steps
                mean = np.sum(rewards_lnr, axis=-1)
                cum_means.append(mean)

                # Convert to torch
                states_lnr = convert_to_tensor(states_lnr)  # (n_envs, horizon, state_dim)
                actions_lnr = convert_to_tensor(actions_lnr)
                next_states_lnr = convert_to_tensor(next_states_lnr)
                rewards_lnr = convert_to_tensor(rewards_lnr[:, :, None])

                # Roll in new data by shifting the batch and appending the new data.
                context_states = torch.cat(
                    (context_states[:, 1:, :, :], states_lnr[:, None, :, :]), dim=1)
                context_actions = torch.cat(
                    (context_actions[:, 1:, :, :], actions_lnr[:, None, :, :]), dim=1)
                context_next_states = torch.cat(
                    (context_next_states[:, 1:, :, :], next_states_lnr[:, None, :, :]), dim=1)
                context_rewards = torch.cat(
                    (context_rewards[:, 1:, :, :], rewards_lnr[:, None, :, :]), dim=1)

            return np.stack(cum_means, axis=1)

        # Start of online evaluation logic
        n_eps = config['online_eval_episodes']
        n_eps_in_context = config['online_eps_in_context']
        n_eval = config['n_eval_envs']
        horizon = config['test_horizon']
        all_means_lnr = []

        envs = []
        for i_eval in range(n_eval):
            traj = eval_trajs[i_eval]
            env = self.create_env(config, traj['goal'], i_eval)
            envs.append(env)

        lnr_controller = TransformerAgent(
            model, batch_size=n_eval, sample=True)
        vec_env = self.create_vec_env(envs)

        cum_means_lnr = _deploy_online_vec(
            vec_env, lnr_controller, n_eps, n_eps_in_context, horizon)

        all_means_lnr = np.array(cum_means_lnr)
        means_lnr = np.mean(all_means_lnr, axis=0)
        sems_lnr = scipy.stats.sem(all_means_lnr, axis=0)

        # Plotting
        for i in range(n_eval):
            plt.plot(all_means_lnr[i], color='blue', alpha=0.2)

        plt.plot(means_lnr, label='Learner')
        plt.fill_between(np.arange(n_eps), means_lnr - sems_lnr,
                         means_lnr + sems_lnr, alpha=0.2)
        plt.legend()
        plt.xlabel('Episodes')
        plt.ylabel('Average Return')
        plt.title(f'Online Evaluation on {n_eval} Envs')


    def offline(self, eval_trajs, model, config, return_envs=False):
        """ Runs each episode separately with offline context. """
        n_eval_envs = config['n_eval_envs']
        n_eval_episodes = config['offline_eval_episodes']

        # Create environments and trajectories
        envs = []
        trajs = []
        for ep_i in range(n_eval_envs):
            env = self.create_env(config, eval_trajs[ep_i]['goal'], ep_i)
            envs.append(env)
            trajs.append(eval_trajs[ep_i])
        batch = {
            'context_states': convert_to_tensor([traj['context_states'] for traj in trajs]),
            'context_actions': convert_to_tensor([traj['context_actions'] for traj in trajs]),
            'context_next_states': convert_to_tensor([traj['context_next_states'] for traj in trajs]),
            'context_rewards': convert_to_tensor([traj['context_rewards'][:, None] for traj in trajs]),
            }

        # Load agents
        print("Running offline evaluations")
        vec_env = self.create_vec_env(envs)
        epsgreedy_agent = TransformerAgent(model, batch_size=n_eval_envs, sample=True)
        epsgreedy_agent_2 = TransformerAgent(model, batch_size=n_eval_envs, temp=1, sample=True)
        greedy_agent = TransformerAgent(model, batch_size=n_eval_envs, sample=False)
        epsgreedy_agent.set_batch(batch)
        epsgreedy_agent_2.set_batch(batch)
        greedy_agent.set_batch(batch)

        # Deploy agents offline
        greedy_returns = []
        epsgreedy_returns = []
        epsgreedy_returns_2 = []
        opt_returns = []
        for _ in range(n_eval_episodes):
            _epsgreedy_obs, _, _, _epsgreedy_returns, _opt_returns_epsgreedy = \
                vec_env.deploy_eval(epsgreedy_agent, return_max_rewards=True)
            _epsgreedy_obs_2, _, _, _epsgreedy_returns_2, _opt_returns_epsgreedy_2 = \
                vec_env.deploy_eval(epsgreedy_agent_2, return_max_rewards=True)
            _greedy_obs, _, _, _greedy_returns, _opt_returns_greedy = \
                vec_env.deploy_eval(greedy_agent, return_max_rewards=True)
            epsgreedy_returns.append(np.sum(_epsgreedy_returns, axis=-1))
            epsgreedy_returns_2.append(np.sum(_epsgreedy_returns_2, axis=-1))
            greedy_returns.append(np.sum(_greedy_returns, axis=-1))
            opt_returns.append(_opt_returns_epsgreedy)
        epsgreedy_returns = np.mean(epsgreedy_returns, axis=0)
        epsgreedy_returns_2 = np.mean(epsgreedy_returns_2, axis=0)
        greedy_returns = np.mean(greedy_returns, axis=0)
        opt_returns = np.mean(opt_returns, axis=0)

        print(f"Epsgreedy returns: {epsgreedy_returns}")
        print(f"Greedy returns: {greedy_returns}")
        print()

        # Plot and return
        baselines = {
            'Opt': opt_returns,
            'Learner (temp=2)': epsgreedy_returns,
            'Learner (temp=1)': epsgreedy_returns_2,
            'Learner (greedy)': greedy_returns,
        }

        if return_envs:
            return baselines, _epsgreedy_obs, envs
        else:
            return baselines, _epsgreedy_obs


    def continual_online(self, eval_trajs, model, config):
        def _deploy_continual_online_vec(vec_env, controller, horizon):
            num_envs = vec_env.num_envs
            batch = {
                'context_states': torch.zeros((num_envs, 0, vec_env.state_dim)).float().to(device),
                'context_actions': torch.zeros((num_envs, 0, vec_env.action_dim)).float().to(device),
                'context_next_states': torch.zeros((num_envs, 0, vec_env.state_dim)).float().to(device),
                'context_rewards': torch.zeros((num_envs, 0, 1)).float().to(device),
            }
            controller.set_batch(batch)
            vec_env.online_batch_size = horizon
            states_lnr, actions_lnr, next_states_lnr, rewards_lnr = vec_env.deploy_eval(
                controller, update_batch_online=True)
            return rewards_lnr # (n_envs, horizon)

        # Start of online evaluation logic
        n_eval_envs = config['n_eval_envs']
        horizon = config['test_horizon']

        envs = []
        for ep_i in range(n_eval_envs):
            traj = eval_trajs[ep_i]
            env = self.create_env(config, traj['goal'], ep_i)
            env.horizon = horizon*30
            envs.append(env)

        lnr_controller = TransformerAgent(
            model, batch_size=n_eval_envs, sample=True)
        vec_env = self.create_vec_env(envs)

        rewards_lnr = _deploy_continual_online_vec(vec_env, lnr_controller, horizon)
        means_lnr = np.mean(rewards_lnr, axis=0)
        sems_lnr = scipy.stats.sem(rewards_lnr, axis=0)

        # Plotting
        for ep_i in range(n_eval_envs):
            plt.plot(rewards_lnr[ep_i], color='blue', alpha=0.2)

        plt.plot(means_lnr, label='Learner')
        plt.fill_between(
            np.arange(rewards_lnr.shape[1]), means_lnr - sems_lnr,
            means_lnr + sems_lnr, alpha=0.2)

        plt.legend()
        plt.xlabel('Steps')
        plt.ylabel('Average Return')
        plt.title(f'Continual online Evaluation on {n_eval_envs} Envs')