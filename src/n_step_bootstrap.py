from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy
import sys

class GreedyPolicy(Policy):

    def __init__(self, nS, nA):
        super(GreedyPolicy, self).__init__()
        self.p = np.zeros((nS, nA))

    def action_prob(self, state:int, action:int) -> float:
        return self.p[state][action]

    def action(self, state:int)-> int:
        return np.argmax(self.p[state])



def on_policy_n_step_td(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    n:int,
    alpha:float,
    initV:np.array
) -> Tuple[np.array]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n: how many steps?
        alpha: learning rate
        initV: initial V values; np array shape of [nS]
    ret:
        V: $v_pi$ function; numpy array shape of [nS]
    """

    T = sys.maxsize
    tau = 0
    sr = [] # list of state, reward tuple pairs


    # loop for each episode
    for episode in trajs:

        while tau != T - 1:

            # for step in episode
            for t in range(len(episode)):
                state = episode[t][0]
                reward = episode[t][2]

                if t < T:
                    sr.append((state, reward))

                    if t == len(episode) - 1:
                        T = t + 1

                tau = t - n + 1

                if tau >= 0:
                    g = np.array([pow(env_spec.gamma, i - tau - 1)
                                    * sr[i][1] for i in range(tau + 1, min(tau + n, T))]).sum()

                    tauState = sr[tau + n - 1][0]

                    if tau + n < T:
                        g = g + pow(env_spec.gamma, n) * initV[tauState]

                    initV[tauState] = initV[tauState] + alpha * (g - initV[tauState])

    V = initV

    return V

def off_policy_n_step_sarsa(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    n:int,
    alpha:float,
    initQ:np.array
) -> Tuple[np.array,Policy]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        n: how many steps?
        alpha: learning rate
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
        policy: $pi_star$; instance of policy class
    """

    # greedy policy
    pi = GreedyPolicy(env_spec.nS, env_spec.nA)

    # loop for each episode
    for episode in trajs:

        sar = []

        # select and store an action
        T = sys.maxsize

        # time step being updated
        tau = 0

        while tau != T - 1:

            # for each step of episode
            for t in range(len(episode)):

                state = episode[t][0]
                action = episode[t][1]
                reward = episode[t][2]
                # state1 = episode[t][3]


                if t < T:
                    # reached last state in episode
                    sar.append((state, action, reward))

                    if t == len(episode) - 1:
                        T = t + 1

                    # time of estimate
                    tau = t - n + 1

                    # reached n-step
                    if tau >= 0:

                        rho = np.array([pi.action_prob(sar[i][0], sar[i][1]) / bpi.action_prob(sar[i][0], sar[i][1])
                                        for i in range(tau + 1, tau + n)]).prod()

                        g = np.array([pow(env_spec.gamma, i - tau - 1) * sar[i][2]
                                      for i in range(tau + 1, min(tau + n, T))]).sum()

                        if tau + n < T:
                            tauState = sar[tau + n - 1][0]
                            tauAction = sar[tau + n - 1][1]

                            g = g + pow(env_spec.gamma, n) * initQ[tauState, tauAction]

                        tauState = sar[tau][0]
                        tauAction = sar[tau][1]
                        initQ[tauState, tauAction] = initQ[tauState, tauAction] + (alpha * rho) * (g - initQ[tauState, tauAction])
                        pi.p[tauState, tauAction] = 1.0

    Q = initQ

    return Q, pi
