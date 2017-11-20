import gym 
import numpy as np 



class Trainer(object):
    def __init__(self, env, agent, args):
        self.args = args
        self.env = env
        self.agent = agent
        self.ob = self.env.reset()

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_logpacs = [],[],[],[],[],[]
        done = False
        total_rew = 0

        for _ in range(self.args.nsteps): # 1 roll-out
            value, action, logpac = self.agent.step(self.ob)

            mb_obs.append(self.ob.copy())
            mb_actions.append(action)
            mb_values.append(value)
            mb_dones.append(done)
            mb_logpacs.append(logpac)
        
            self.ob, reward, done, _ = self.env.step(action)

            #####################
            self.env.render()
            #####################

            total_rew += reward
            mb_rewards.append((reward+8) / 8) ###### For now ............

        mb_obs = np.asarray(mb_obs, dtype=self.ob.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).reshape((-1, 1))
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32).reshape((-1, 1))
        mb_logpacs = np.array(mb_logpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).reshape((-1, 1))

        last_value = self.agent.get_value(self.ob)

        '''
        # discount / boostrap off value
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0   

        for t in reversed(range(self.args.nsteps)):
            if t == self.args.nsteps - 1:
                nextnonterminal = 1.0 - done
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.args.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.args.gamma * self.args.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        '''

        discounted_r = []
        for r in mb_rewards[::-1]:
            last_value = r + self.args.gamma * last_value
            discounted_r.append(last_value)
        discounted_r.reverse()
        discounted_r = np.asarray(discounted_r, dtype=np.float32)

        return mb_obs, mb_actions, discounted_r, mb_logpacs, total_rew
        # return mb_obs, mb_actions, mb_values, mb_dones, mb_logpacs, mb_returns, mb_rewards, discounted_r


    def learn(self):
        # Number of examples in one big batch
        nbatch = self.args.nenvs * self.args.nsteps
        nbatch_train = nbatch // self.args.nminibatches

        # Total number of steps to run simulation
        total_timesteps = self.args.num_timesteps
        # Number of times to run optimization
        nupdates = int(total_timesteps // nbatch)

        for update in range(1, nupdates+1):
            assert nbatch % self.args.nminibatches == 0

            '''
            ####################################
            # Adaptive clip-range and learning-rate decaying
            frac = 1.0 - (update - 1.0) / nupdates
            clip_range_now = self.args.clip_range(frac)
            lr_now = self.args.lr_schedule(frac)
            #####################################
            '''

            self.ob = self.env.reset()
            ep_r = 0

            # Specific for Pendulum, to reset the episode
            for _ in range(7):
                obs, actions, discounted_r, logpacs, total_rew = self.run()
                ep_r += total_rew

                inds = np.arange(nbatch)

                for _ in range(self.args.num_update_epochs):
                    np.random.shuffle(inds)
                    # Per mini-batches in one roll-out
                    for start in range(0, nbatch, nbatch_train):
                        end = start + nbatch_train
                        batch_inds = inds[start : end]
                        
                        self.agent.update(obs[batch_inds], actions[batch_inds], \
                                          discounted_r[batch_inds], logpacs[batch_inds])

            print('Ep: %i' % update, "|Ep_r: %i" % ep_r) 


def test():
    from params import Pendulum_Params
    from ppo import PPO

    env = gym.make('Pendulum-v0').unwrapped
    params = Pendulum_Params()
    ppo = PPO(env, params)
    trainer = Trainer(env, ppo, params)

    trainer.learn()
    print('Learn success')


if __name__ == "__main__":
    test()