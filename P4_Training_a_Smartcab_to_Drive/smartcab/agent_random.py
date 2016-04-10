import random
import numpy as np
from collections import OrderedDict
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


class LearningAgent(Agent):
    """An basic agent that moves randomly in the grid world."""

    def __init__(self, env):
        # set self.env = env, state = None, next_waypoint = None
        # set default color
        super(LearningAgent, self).__init__(env)

        # override color
        self.color = 'red'

        # simple route planner to get next_waypoint
        self.planner = RoutePlanner(self.env, self)
        self.next_waypoint = random.choice(Environment.valid_actions)

        # TODO: Initialize any additional variables here
        self.rewards = []
        self.rewards_dict = OrderedDict()
        self.deadlines = []

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.rewards = []
        self.deadlines.append(self.env.agent_states[self]['deadline'])

        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t, trial):
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: select random action
        action = self.next_waypoint
        reward = self.env.act(self, action)
        self.rewards.append(reward)
        self.rewards_dict[str(trial)] = self.rewards
        self.next_waypoint = random.choice(Environment.valid_actions)

        # print "LA.update (t={}) || dl = {} || inputs = {} || act = {} || r = {}". \
        #     format(t + 1, deadline, inputs, action, reward)


def run_stats(a, e, showplot=True):
    """produce simulation statistics and graphs for (n_trials)"""

    # list of the total steps taken for each tiral
    steps_per_trial = [len(rewards) for _, rewards in a.rewards_dict.iteritems()]

    df_rows = ['steps to dest.', 'deadline']
    df = pd.DataFrame([steps_per_trial, a.deadlines], index=df_rows)
    df = df.T

    dest_reached_before_deadline = str(sum(df['steps to dest.'] < df['deadline']))
    title_str = 'Random AI: Reached dest. before deadline: ' + dest_reached_before_deadline + \
                ' out of ' + str(len(steps_per_trial)) + ' trials'

    # set matplotlib parameters
    rc('axes', linewidth=2)
    rc('font', weight='bold')

    if showplot:
        # fig 1:  plot cumulated rewards
        plt.figure()
        plt.set_cmap('viridis')
        for trial_no, rewards in a.rewards_dict.iteritems():
            cum_rewards = np.cumsum(rewards)
            plt.plot(cum_rewards, label=('trial ' + trial_no), linewidth=2)
            plt.xlabel('Step', fontweight='bold')
            plt.title("Cumulated rewards for separate trials", fontweight='bold')
            plt.legend(loc='lower right')

        # fig 2: steps to dest. vs deadline
        df.plot(kind='bar')
        plt.ylim(0, df.values.max()+5)
        plt.title(title_str, fontweight='bold')
        plt.xlabel('Trial No.', fontweight='bold')
        plt.ylabel('Steps taken to destination', fontweight='bold')

    pd.set_option('precision', 1)
    print(df.describe())
    print('--------------------------------')
    print(title_str)
    print('Hard time limit: %d' % e.hard_time_limit)

    return dest_reached_before_deadline


def run():
    """Run the simulation for (n_trials)"""

    # create environment (also adds some dummy traffic)
    e = Environment()

    # create primary agent and simulator without enforcing deadline
    a = e.create_agent(LearningAgent)
    e.set_primary_agent(a, enforce_deadline=False)
    sim = Simulator(e, update_delay=.001, display=False)

    # run simulation x 10 + plotting and analysis
    n_trials = 10
    print('===================================================')
    print('Total number of trials: %d' % n_trials)
    print('--------------------------------')
    sim.run(n_trials=n_trials)
    run_stats(a, e)
    print('===================================================')

    # show plots
    plt.show()


if __name__ == '__main__':
    run()
