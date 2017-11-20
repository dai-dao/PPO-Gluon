import gym 
import numpy as np 



class Trainer(object):
    def __init__(self, env, agent, args):
        self.args = args
        self.env = env
        self.agent = agent


    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_logpacs = [],[],[],[],[],[]
        ob = self.env.reset()
        done = False

        for _ in range(self.args.nsteps):
            value, action, logpac = self.agent.step(ob)

            mb_obs.append(ob)
            mb_actions.append(action)
            mb_values.append(value)
            mb_dones.append(done)
            mb_logpacs.append(logpac)
        
            ob, reward, done, _ = self.env.step([action])
        
            mb_rewards.append(reward)

        mb_obs = np.asarray(mb_obs, dtype=ob.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).reshape((-1, 1))
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32).reshape((-1, 1))
        mb_logpacs = np.asarray(mb_logpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).reshape((-1, 1))

        last_value = self.agent.get_value(ob)
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

        return mb_obs, mb_actions, mb_values, mb_dones, mb_logpacs, mb_returns


    def learn(self):
        obs, actions, values, dones, logpacs, returns = self.run()

        


def test():
    from params import Params
    from ppo import PPO

    env = gym.make('Pendulum-v0')
    params = Params()
    ppo = PPO(env, params)
    trainer = Trainer(env, ppo, params)

    print('Init success')

    trainer.train()
    print('Train success')


if __name__ == "__main__":
    test()