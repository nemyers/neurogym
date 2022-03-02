#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import neurogym as ngym
from neurogym import spaces


class DelayMatchCategoryMod(ngym.TrialEnv):
    """Delayed match-to-category task.

    A sample stimulus is shown during the sample period. The stimulus is
    characterized by a one-dimensional variable, such as its orientation
    between 0 and 360 degree. This one-dimensional variable is separated
    into two categories (for example, 0-180 degree and 180-360 degree).
    After a delay period, a test stimulus is shown. The agent needs to
    determine whether the sample and the test stimuli belong to the same
    category, and report that decision during the decision period.
    """
    metadata = {
        'paper_link': 'https://www.nature.com/articles/nature05078',
        'paper_name': '''Experience-dependent representation
        of visual categories in parietal cortex''',
        'tags': ['perceptual', 'working memory', 'two-alternative',
                 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0,
                 dim_ring=2,stim_range=0):
        super().__init__(dt=dt)
        self.choices = ['match', 'non-match']  # match, non-match

        self.sigma = sigma / np.sqrt(self.dt)  # Input noise
        self.stim_range = stim_range # Stimulus Orientation Jitter

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': 500,
            'sample': 650,
            'first_delay': 1000,
            'test': 650}

        if timing:
            self.timing.update(timing)

        self.abort = False

        self.theta = np.linspace(0, 2 * np.pi, dim_ring + 1)[:-1]

        name = {'fixation': 0, 'stimulus': range(1, dim_ring + 1)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1 + dim_ring,), dtype=np.float32, name=name)

        name = {'fixation': 0, 'match': 1, 'non-match': 2}
        self.action_space = spaces.Discrete(3, name=name)

    def _new_trial(self, **kwargs):
        # Trial info
        trial = {
            'ground_truth': self.rng.choice(self.choices),
            'sample_category': self.rng.choice([0, 1]),
        }
        trial.update(**kwargs)

        ground_truth = trial['ground_truth']
        sample_category = trial['sample_category']
        if ground_truth == 'match':
            test_category = sample_category
        else:
            test_category = 1 - sample_category

        #sample_theta = (sample_category + self.rng.rand()) * np.pi
        #test_theta = (test_category + self.rng.rand()) * np.pi
        sample_theta = (sample_category + self.rng.rand()*self.stim_range - 0.0) * np.pi
        test_theta = (test_category + self.rng.rand()*self.stim_range - 0.0) * np.pi
        trial.update({'sample_theta': sample_theta, 'test_theta': test_theta})

        #stim_sample = np.cos(self.theta - sample_theta) * 0.5 + 0.5
        #stim_test = np.cos(self.theta - test_theta) * 0.5 + 0.5
        
        sigma = 2*(43.2*np.pi/180)**2
        a     = 0.8
        
        adiff = (self.theta-sample_theta+3*np.pi)%(2*np.pi)-np.pi
        stim_sample = a*np.exp(-(adiff)**2/sigma)
        adiff = (self.theta-test_theta+3*np.pi)%(2*np.pi)-np.pi
        stim_test   = a*np.exp(-(adiff)**2/sigma)

        stim_fixation = 0.05
        # Periods
        periods = ['fixation', 'sample', 'first_delay', 'test']
        self.add_period(periods)

        self.add_ob(stim_fixation, where='fixation')
        self.set_ob(0, 'test', where='fixation')
        self.add_ob(stim_sample, 'sample', where='stimulus')
        self.add_ob(stim_test, 'test', where='stimulus')
        # self.add_randn(0, self.sigma, ['sample', 'test'], where='stimulus')
        self.add_randn(0, self.sigma, ['first_delay', 'test'], where='stimulus')

        self.set_groundtruth(self.action_space.name[ground_truth], 'test')

        return trial

    def _step(self, action, **kwargs):
        new_trial = False

        ob = self.ob_now
        gt = self.gt_now

        reward = 0
        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']
        elif self.in_period('test'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward = self.rewards['correct']
                    self.performance = 1
                else:
                    reward = self.rewards['fail']

        return ob, reward, False, {'new_trial': new_trial, 'gt': gt}
