from __future__ import division
import itertools
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
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, elseif=None):
        # set self.env = env, state = None, next_waypoint = None
        # set default color
        super(LearningAgent, self).__init__(env)

        # override color
        self.color = 'red'

        # simple route planner to get next_waypoint
        self.planner = RoutePlanner(self.env, self)

        # TODO: Initialize any additional variables here
        self.state = None
        self.new_state = None
        self.inputs = ()
        self.rewards = []
        self.rewards_dict = OrderedDict()
        self.deadlines = []

        self.q_gamma = 0.85
        self.q_alpha = 0.15
        self.q_epsilon = 0.15

        self.q_states = [('green', 'red'),
                         ('None', 'forward', 'right', 'left'),
                         (True, False)]

        self.q_states = list(itertools.product(*self.q_states))
        self.q_actions = [True, False]

        # create q-table with (state, action) pairs
        # initialize q-table with random values
        index = pd.MultiIndex.from_tuples(self.q_states)
        self.q_table = pd.DataFrame(np.random.rand(len(self.q_actions), len(self.q_states)).T,
                                    columns=self.q_actions, index=index)

    def reset(self, destination=None):

        # TODO: Prepare for a new trip; reset any variables here, if required
        self.planner.route_to(destination)
        self.rewards = []
        self.deadlines.append(self.env.agent_states[self]['deadline'])

    def update(self, t, trial):

        deadline = self.env.get_deadline(self)
        inputs = self.env.sense(self)

        # TODO: Update state
        any_car_nearby = any([inputs['left'], inputs['right'], inputs['oncoming']])
        self.state = (str(inputs['light']), str(self.next_waypoint), any_car_nearby)

        # TODO: Select action according to your policy
        action = None

        # epsilon greedy action selection
        if self.q_epsilon < np.random.random_sample():
            action_ok = self.q_table.loc[self.state].argmax()
        else:
            action_ok = np.random.choice(self.q_actions)

        if action_ok:
            action = self.next_waypoint

        # collect reward r
        reward = self.env.act(self, action)
        self.rewards.append(reward)
        self.rewards_dict[str(trial)] = self.rewards

        # plan the next move
        self.next_waypoint = self.planner.next_waypoint()

        # sense the new state s'
        inputs = self.env.sense(self)
        any_car_nearby = any([inputs['left'], inputs['right'], inputs['oncoming']])
        self.new_state = (str(inputs['light']), str(self.next_waypoint), any_car_nearby)

        # TODO: Learn policy based on state, action, reward
        term_A = (1 - self.q_alpha) * self.q_table.loc[self.state][action_ok]
        term_B = self.q_alpha * (reward + self.q_gamma * self.q_table.loc[self.new_state].max())

        self.q_table.loc[self.state][action_ok] = term_A + term_B

        # self.q_alpha = 1 / (t + 1)
        # print "LA.update (t={}) || dl = {} || inputs = {} || act = {} || r = {}" \
        # .format(t + 1, deadline, inputs, action, reward)


def run_stats(a, e, showplot=True):
    """produce simulation statistics and graphs for (n_trials)"""

    # list of the total steps taken for each tiral
    steps_per_trial = [len(rewards) for _, rewards in a.rewards_dict.iteritems()]

    df_rows = ['steps to dest.', 'deadline']
    df = pd.DataFrame([steps_per_trial, a.deadlines], index=df_rows)
    df = df.T

    dest_reached_before_deadline = str(sum(df['steps to dest.'] < df['deadline']))
    title_str = 'LearningAgent reached dest. before deadline: ' + dest_reached_before_deadline + \
                ' out of ' + str(len(steps_per_trial)) + ' trials'

    # set matplotlib parameters
    rc('axes', linewidth=2)
    rc('font', weight='bold')

    if showplot:
        # fig 1:  plot cumulated rewards
        plt.figure()
        for trial_no, rewards in a.rewards_dict.iteritems():
            cum_rewards = np.cumsum(rewards)
            plt.plot(cum_rewards, label=('trial ' + trial_no), linewidth=2)
            plt.xlabel('Step', fontweight='bold')
            plt.title("Cumulated rewards for separate trials", fontweight='bold')
            plt.legend(loc='lower right')
        plt.viridis()

        # fig 2: steps to dest. vs deadline
        df.plot(kind='bar')
        plt.ylim(0, df.values.max() + 5)
        plt.title(title_str, fontweight='bold')
        plt.xlabel('Trial No.', fontweight='bold')
        plt.ylabel('Steps taken to destination', fontweight='bold')

    # pd.set_option('precision', 1)
    # print(df.describe())
    # print('Hard time limit: %d' % e.hard_time_limit)

    print(title_str)

    return dest_reached_before_deadline


def run_exp(a, e, sim, n_exp=1, n_trials_test=10, n_trials_train=100, showplot=False, show_q_table=False):
    ntimes_dest_reached_before_deadline = []

    for exp in range(n_exp):
        # training - run simulation x 100 + plotting and analysis
        print('==========================================================')
        print('Experiment number: %d\n' % exp)
        print('Total number of trials: %d' % n_trials_train)
        sim.run(n_trials=n_trials_train)
        run_stats(a, e, showplot=False)

        # testing - run simulation x 10 + plotting and analysis
        a.rewards_dict = OrderedDict()
        a.deadlines = []

        # fully greedy - solely use the q-table to select actions
        a.q_epsilon = 0

        print('\nTotal number of trials: %d' % n_trials_test)
        sim.run(n_trials=n_trials_test)
        ntimes_dest_reached_before_deadline.append(run_stats(a, e, showplot=showplot))

        # reset deadlines and q_table after each experiment
        a.deadlines = []

        if show_q_table:
            print('==========================================================')
            a.q_table.columns = ['action_ok', 'action_not_ok']
            pd.set_option('display.float_format', lambda x: '%.2f' % x)
            print(a.q_table)
            print('==========================================================')

        a.q_table = pd.DataFrame(0, index=a.q_table.index, columns=a.q_table.columns)

    if n_exp > 1:
        fig, ax = plt.subplots(1, 1)

        ntimes_dest_reached_before_deadline = \
            np.array(ntimes_dest_reached_before_deadline).astype(float)

        proba_dest_reached_before_deadline = \
            100 * sum(ntimes_dest_reached_before_deadline) / (n_trials_test * n_exp)

        bins = range(1, 11)
        ax.hist(ntimes_dest_reached_before_deadline, bins=bins, alpha=0.75, align='right')
        plt.xticks(range(1, 11))
        plt.xlabel('No. of times reached dest. before deadline (x out of 10)')
        plt.ylabel('No. of experiments')

        print('==========================================================')
        print('For %d expts of (train: %d trials, test: %d trials),'
              % (n_exp, n_trials_train, n_trials_test))
        print('agent reached dest. before deadline with a probability of\n%.1f %% on testing trials.'
              % proba_dest_reached_before_deadline)
        print('==========================================================')

        return proba_dest_reached_before_deadline


def run():
    """run the simulation for (n_trials)"""

    # create environment (also adds some dummy traffic)
    e = Environment()

    # create primary agent and simulator object
    a = e.create_agent(LearningAgent)
    e.set_primary_agent(a, enforce_deadline=True)
    sim = Simulator(e, update_delay=.001, display=False)

    # run learning experiments x n_exp
    n_exp = 1
    n_trials_train = 100
    n_trials_test = 10

    a.q_gamma = 0.85
    a.q_alpha = 0.20
    a.q_epsilon = 0.20
    showplot = True
    show_q_table = True

    tuning_alpha = False
    q_alphas = [0.1, 0.2, 0.3, 0.4, 0.5]

    tuning_epsilon = False
    q_epsilons = [0.1, 0.3, 0.5, 0.7, 1]

    # run independent experiments n_exp x (train, test)
    if not (tuning_alpha or tuning_epsilon):
        run_exp(a, e, sim, n_exp, n_trials_test, n_trials_train, showplot, show_q_table)

    # tuning learning rate a.q_alpha (very long run time)
    # total number of trials = len(q_alphas) * n_exp * (n_trials_test + n_trials_train)
    if tuning_alpha:
        probs_alpha = []
        for q_alpha in q_alphas:
            a.q_alpha = q_alpha
            prob_alpha = run_exp(a, e, sim, n_exp, n_trials_test, n_trials_train)
            probs_alpha.append(prob_alpha)

        print('==========================================================')
        for q_alpha, prob_alpha in zip(q_alphas, probs_alpha):
            print('q_alpha: %.2f  prob. of success: %.2f' % (q_alpha, prob_alpha))
        print('==========================================================')

        plt.figure()
        plt.plot(q_alphas, probs_alpha, marker='s')
        plt.xlabel('Alpha', fontweight='bold')
        plt.ylabel('Prob. of reaching dest. before deadline', fontweight='bold')
        plt.title('Optimal alpha is %.2f' % q_alphas[np.argmax(probs_alpha)], fontweight='bold')
        plt.xlim(min(q_alphas) - 0.05, max(q_alphas) + 0.05)
        plt.ylim(min(probs_alpha) - 5, max(probs_alpha) + 5)

    if tuning_epsilon:
        probs_epsilon = []
        for q_epsilon in q_epsilons:
            a.q_epsilon = q_epsilon
            probs_epsilon.append(run_exp(a, e, sim, n_exp, n_trials_test, n_trials_train))

        plt.figure()
        plt.plot(q_epsilons, probs_epsilon, marker='s')
        plt.xlabel('Epsilon', fontweight='bold')
        plt.ylabel('Prob. of reaching dest. before deadline', fontweight='bold')
        plt.title('Optimal epsilon is %.2f' % q_epsilons[np.argmax(probs_epsilon)], fontweight='bold')
        plt.xlim(min(q_epsilons) - 0.05, max(q_epsilons) + 0.05)
        plt.ylim(min(probs_epsilon) - 5, max(probs_epsilon) + 5)

    # show plot
    plt.show()


if __name__ == '__main__':
    run()
