"""This file contains measure that we may take into account to a later amount in time where implementations are
prepared """
import sqlite3
import numpy as np
import matplotlib.pyplot as plt


DB_FILE = '/home/anjakoller/tezos_dataextraction_merged_alltables.db'


def Group_negentropy(x_i):
    if x_i == 0:
        return 0
    else:
        return x_i * np.log(x_i)


def H(x):
    n = len(x)
    entropy = 0.0
    sum = 0.0
    for x_i in x:  # work on all x[i]
        sum += x_i
        group_negentropy = Group_negentropy(x_i)  # x_i*log(x_i)
        entropy += group_negentropy
    return -entropy


def compute_theil_index(arr):
    """Computes theil index according to the formula from wiki: https://de.wikipedia.org/wiki/Theil-Index,
    see: https://stackoverflow.com/questions/20279458/implementation-of-theil-inequality-index-in-python
    see also: https://www.classicistranieri.com/de/articles/t/h/e/Theil-Index_c341.html"""
    print('x', arr)
    n = len(arr)
    maximum_entropy = np.log(n)
    actual_entropy = H(arr)
    redundancy = maximum_entropy - actual_entropy
    inequality = 1 - np.exp(-redundancy)
    return redundancy, inequality


def get_lorenz(arr):
    arr = np.asarray(arr)
    # first sort array from lowest to highest
    np.sort(arr)
    # this divides the prefix sum by the total sum
    # this ensures all the values are between 0 and 1.0
    scaled_prefix_sum = arr.cumsum() / arr.sum()
    # this prepends the 0 value (because 0% of all people have 0% of all wealth)
    return np.insert(scaled_prefix_sum, 0, 0)


def compute_reward_era_baker_per_cycle(start, end):
    avg_reward_per_cycle = []
    reward_list_tuple = cur.execute('SELECT reward FROM cyclerewards where cycle>=? AND cycle<=?',
                                    (start, end)).fetchall()
    for reward in reward_list_tuple:
        avg_reward_per_cycle.append(reward[0])
    return avg_reward_per_cycle


def compute_proportion_snapshot_rolls(start, end):
    # TODO: corresponds to the fairness plot
    """ Computed the proportion of rolls in a snapshot compared to all the rolls (for each baker) """
    proportion_rolls_list = []
    return proportion_rolls_list


def compute_proportion_rewards_all_bakers_per_cycle(start, end):
    """Computes a list for each baker with the proportion of the rolls compared to the total rolls in the respective
    cycles """
    # TODO: implement this, corresponds to fairness plot
    proportion_rolls_list = []
    return proportion_rolls_list


def compute_base_reward_propotion():
    # TODO: implement the avg/base proportion expected fairness
    return 1


def compute_base_rolls_proportion():
    # TODO: implement avg/base proportion expected fairness
    return 1

    def error_if_not_in_range01(value):
        if (value <= 0) or (value > 1):
            raise Exception(str(value) + ' is not in [0,1)!')


def plot_lorenz_curve(arr):
    """The Lorenz curve is a common graphical method to represent the degree of income inequality. It plots the
    cumulative share of income (y axis) earned by the poorest x percentages of the population,for all values of x """
    # x axis: percentage of participants
    y_data = get_lorenz(arr)  # percentage of share of reward
    plt.plot(np.linspace(0.0, 1.0, get_lorenz(arr).size), y_data, label='Lorenz curve')
    # plot the straight line perfect equality curve
    plt.plot([0, 1], [0, 1])
    plt.title('Lorenz Curve')
    plt.xlabel('Percentage of Participants')
    plt.ylabel('Percentage of Income share')
    plt.savefig('images/Lorenz_curve')
    plt.close()


def plot_fairness_comparison_rewards_rolls(start, end, alpha):
    x1_data = list(range(start, end + 1))
    y1_data = compute_proportion_snapshot_rolls(start, end)
    # TODO: probably shifted by t-7
    x2_data = list(range(start, end + 1))
    y2_data = compute_proportion_rewards_all_bakers_per_cycle(start, end)
    plt.plot(x1_data, y1_data, label='Rolls')
    plt.plot(x2_data, y2_data, label='Rewards')
    base_reward = compute_base_reward_propotion()
    base_roll = compute_base_rolls_proportion()
    # TODO: eventually only one line then with two sublines at plus/minus alpha
    plt.axhline(base_reward, 0, 1, label='Baseline proportion reward', color='red')
    plt.axhline(base_roll, 0, 1, label='Baseline proportion rolls', color='green')
    plt.title('Fairness Comparison rewards and rolls proportion from' + str(start) + ' to ' + str(end))
    plt.xlabel('Cycle')
    plt.ylabel('Proportion of total rewards/rolls')
    plt.savefig('images/Fairness_comparison_rewards_rolls_' + str(start) + '_' + str(end) + '.png')
    plt.close()


# TODO: How much did a baker earn per cycle, snapshot t-7 corresponds to cycle t --> rewards per baker per cycle
# then compare with snapshot t-7
# def plot_snapshots_rolls_gini_index(start, end):
# take only some cycle's snapshots to look at individual sections
#    gini_indexes_all_bakers_snapshot = compute_gini_snapshot_rolls(start, end)
#    x_data = list(range(start, end + 1))
#    y_data = gini_indexes_all_bakers_snapshot
#    plt.plot(x_data, y_data)
#    plt.title('Gini indexes Snapshot rolls (reward) from cycles ' + str(start) + ' to ' + str(end))
#    plt.xlabel('Snapshots Cycles')
#    plt.ylabel('Gini index')
#    plt.ylim(0.6, 0.95)  # make it the same scale as the plots for rewards
#    plt.savefig('snapshots/Snapshot_rolls_cycle_' + str(start) + '_to_' + str(end) + '_gini_index.png')
#    plt.close()


if __name__ == '__main__':
    # Setup db
    con = sqlite3.connect(DB_FILE)  # attributes: cycle, baker, fee, reward, deposit, blocks (merged db)
    cur = con.cursor()

    # Lorenz curve
    rewards = compute_reward_era_baker_per_cycle(0, 404)
    # TODO: look at it only at a specific cycle for all the bakers -> or at 3 different cycles for example
    plot_lorenz_curve(rewards)  # compute lorenz curve with reward per baker
    # compute lorenz curve with income_fees() # TODO: use income table for this

    # Theil index
    # Theil-Index is the "contribution" of the subgroup to the inequality of the whole group
    # below needs to be done with subgroups
    redundancy, inequality = compute_theil_index(rewards)
    print('Theil index redundancy', redundancy)
    print('Theil index inequality', inequality)

    # Snapshot rolls gini index plots
    # TODO: check this does not seem to give a reasonable gini for the rewards -> DEBUG -> TOO many snapshots
    # for start, end in zip(start_cycles, end_cycles):
    #    plot_snapshots_rolls_gini_index(start, end)

    # TODO: fairness measure -> in a cycle compare proportion of rolls and proportion of rewards -> define an "ok
    #  band" and a stochastic variation which is ok
    # TODO: below
    # plot_fairness_comparison_rewards_rolls(250, 251, 0.03)  # look at cycle 250 and have a variation of 3% tolerance


# Note: The right to publish a block in cycle n is assigned to a randomly selected roll in a randomly selected
# roll snapshot from cycle n - PRESERVED_CYCLES — 2 (which is n-7)
# roll n-7 must be somehow coherent with rewards n (rewards are payed out 7 cycles after snapshot is taken)