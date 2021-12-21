import sqlite3
import numpy as np
import matplotlib.pyplot as plt

DB_FILE = '/home/anjakoller/tezos_dataextraction_merged_alltables.db'


def compute_fractions(start, end, baker):
    """Currently works for 1 specific baker"""
    rewards = []
    rews = cur.execute('select total_income from income_table where address '
                       '="%s" and cycle >= %s and cycle <= %s' % (baker,
                                                                  start, end)).fetchall()
    for init_rew in rews:
        rewards.append(init_rew[0])
    n = len(rewards)
    initial_total = cur.execute('select sum(total_income) from income_table where cycle = 0').fetchall()[0][0]
    initial_reward = cur.execute('select total_income from income_table where cycle = 0 and '
                                 'address="%s"' % baker).fetchall()[0][0]

    total_rewards = []  # total_rewards at cycle x for all bakers
    for c in range(start, end + 1):
        rew_c = cur.execute('select sum(total_income) from income_table where cycle = %s' % c).fetchall()[0][0]
        total_rewards.append(rew_c)

    # Fractions: array of length n, where each entry divides rewards[i]/total_rewards[i] for i = 1 to n
    fractions = []
    for c in range(0, n):
        frac_c = rewards[c] / total_rewards[c]
        fractions.append(frac_c)
    expected = [initial_reward / initial_total] * n
    return fractions, expected


def compute_difference(actual, expected):
    """Computes difference array of the actual and expected values"""
    differences = []
    n = len(actual)
    for c in range(0, n):
        diff_c = (actual[c] - expected[c])
        differences.append(diff_c)
    return differences


def plot_expectational_fairness_difference(start, end, baker):
    x_data = list((range(start, end + 1)))
    actual, expected = compute_fractions(start, end, baker)
    y_data = compute_difference(actual, expected)
    plt.plot(x_data, y_data)
    plt.xlabel('Cycle')
    plt.ylabel('Difference actual and expected reward fraction')
    plt.title('Exp. Fairness Diff. Baker ' + baker)
    plt.savefig('images/expectational_fairness/' + str(start) + '_' + str(end) + '_baker_' + baker + '.png')
    plt.close()


def compute_fractions(start, end, baker):
    """Currently works for 1 specific baker"""
    rewards = []
    rews = cur.execute('select total_income from income_table where address '
                       '="%s" and cycle >= %s and cycle <= %s' % (baker,
                                                                  start, end)).fetchall()
    for init_rew in rews:
        rewards.append(init_rew[0])
    n = len(rewards)
    initial_total = cur.execute('select sum(total_income) from income_table where cycle = 0').fetchall()[0][0]
    initial_reward = cur.execute('select total_income from income_table where cycle = 0 and '
                                 'address="%s"' % baker).fetchall()[0][0]

    total_rewards = []  # total_rewards at cycle x for all bakers
    for c in range(start, end + 1):
        rew_c = cur.execute('select sum(total_income) from income_table where cycle = %s' % c).fetchall()[0][0]
        total_rewards.append(rew_c)

    # Fractions: array of length n, where each entry divides rewards[i]/total_rewards[i] for i = 1 to n
    fractions = []
    for c in range(0, n):
        frac_c = rewards[c] / total_rewards[c]
        fractions.append(frac_c)
    expected = [initial_reward / initial_total] * n
    return fractions, expected


def plot_expectational_fairness(start, end, baker):
    """expectational fairness plot just for the specific baker for all cycles from start to end"""
    x_data = list((range(start, end + 1)))
    # TODO: redo this
    y_data, y2_data = compute_fractions(start, end, baker)
    plt.plot(x_data, y_data, label='Fraction of resource actual')
    plt.plot(x_data, y2_data, label='Fraction of initial resource (expected)')
    plt.legend()
    plt.xlabel('Cycle')
    plt.ylabel('Fraction of reward')
    plt.title('Expectational Fairness baker' + baker)
    plt.savefig('images/expectational_fairness/expectational_fairness_onebaker_' + str(start) + '_' + str(
        end) + 'baker_' + baker + '.png')
    plt.close()


def compute_fairness_percentage_average_one_cycle(cycle):
    percentages = compute_fairness_percentage_one_cycle(cycle)
    avg_percent = np.mean(percentages)  # mean of the percentages over all bakers for a cycle
    return avg_percent


def plot_expecational_fairness_all_bakers_overview(start, end, lowest_x, lowest_x2, highest_x):
    y1_highest, y1_lowest = compute_fairness_highest_x_percent_all_cycles(start, end, highest_x)
    y2_highest, y2_lowest = compute_fairness_highest_x_percent_all_cycles(start, end, lowest_x)
    y3_highest, y3_lowest = compute_fairness_highest_x_percent_all_cycles(start, end, lowest_x2)
    x_data = list((range(start, end)))
    y1_data = y1_highest
    plt.plot(x_data, y1_data, '.', color='black', label='highest ' + str(highest_x) + ' percent')
    y2_data = y2_lowest
    # plt.plot(x_data, y2_data, '.', color='green', label='highest ' + str(lowest_x) + ' percent')
    y3_data = y3_lowest
    plt.plot(x_data, y3_data, '.', color='blue', label='lowest ' + str(lowest_x2) + ' percent')
    y4_data = compute_fairness_percentages_average_all_cycles(start, end)
    plt.plot(x_data, y4_data, '.', color='red', label='average')
    plt.legend()
    plt.xlabel('Cycle')
    plt.ylabel('Absolute reward/Expected reward')
    plt.title('Expectational Fairness from cycle ' + str(start) + ' to ' + str(end))
    plt.savefig('images/expectational_fairness/expectational_fairness_cycles_' + str(start) + '_' + str(
        end) + '_overview' + '.png')
    plt.close()


def compute_fairness_percentage_one_cycle(cycle):
    fractions, expected = compute_fraction_all_bakers_one_cycle(cycle)
    n = len(fractions)  # number of bakers
    percentages = []  # contains percentages for all the bakers in a specific cycle
    for p in range(0, n):
        if expected[p] == 0:
            percent_p = 0
        else:
            percent_p = fractions[p] / expected[p]
        percentages.append(percent_p)
    return percentages


def compute_fairness_highest_x_percent_all_cycles(start, end, x):
    percentages_highest = []
    percentages_lowest = []
    num_cycles = end - start
    for c in range(0, num_cycles):
        highest_x, lowest_x = compute_fairness_percentage_one_cycle_x(c, x)
        percentages_highest.append(highest_x)
        percentages_lowest.append(lowest_x)
    return percentages_highest, percentages_lowest



def compute_fairness_percentage_one_cycle_x(cycle, x):
    percentages = compute_fairness_percentage_one_cycle(cycle)
    x_percent = int(np.ceil(len(percentages) / (x * 100)))
    percentages = np.asarray(percentages)
    highest_x_percentages = np.sort(percentages)[(len(percentages) - x_percent):]
    highest_x_percent = np.mean(highest_x_percentages)  # mean over the highest 90% of the bakers
    lowest_x_percentages = np.sort(percentages)[:(len(percentages) - x_percent)]
    lowest_x_percent = np.mean(lowest_x_percentages)
    return highest_x_percent, lowest_x_percent

def get_all_active_bakers():
    """returns the 8 bakers that are active in all cycle"""
    bakers = []
    bakers_list = cur.execute(
        'SELECT address from income_table where cycle=0').fetchall()
    for b in bakers_list:
        bakers.append(b[0])
    return bakers


def get_baker_at_cycle(cycle):
    """return an array of addresses of all bakers at cycle cycle"""
    bakers = []
    baker = cur.execute('select address from income_table where cycle= %s' % cycle).fetchall()
    for b in range(0, len(baker)):
        bakers.append(baker[b][0])
    return bakers


def get_income_at_cycle(cycle):
    """return an array of incomes of all bakers at cycle cycle"""
    rewards = []  # list with initial income for all the bakers
    reward = cur.execute('select total_income from income_table where cycle = %s' % cycle).fetchall()
    for c in range(0, len(reward)):
        rewards.append(reward[c][0])
    return rewards


def get_bakers_initial_rewards_at_cycle(cycle):
    # initialize initial bakers and rewards array
    bakers = get_baker_at_cycle(0)
    rewards = get_income_at_cycle(0)
    initial_totals = [cur.execute('SELECT SUM(total_income) from income_table where cycle=0').fetchall()[0][0]] * 8

    # for every cycle from 0 to cycle, check if baker already exist, if not add it and add its initial_reward
    for c in range(1, cycle + 1):
        new_bakers = get_baker_at_cycle(c)
        for baker in new_bakers:
            if not baker in bakers:
                bakers.append(baker)
                baker_reward = cur.execute(
                    'SELECT total_income from income_table where address="%s" and cycle=%s' % (baker, c)).fetchall()[0][
                    0]
                rewards.append(baker_reward)
                c_total = cur.execute('SELECT SUM(total_income) from income_table where cycle=%s' % c).fetchall()[0][0]
                initial_totals.append(c_total)
    return bakers, rewards, initial_totals


def compute_fraction_all_bakers_one_cycle(cycle):
    initial_bakers, initial_rewards, initial_totals = get_bakers_initial_rewards_at_cycle(cycle)
    initial_bakers = tuple(initial_bakers)  # convert to tuple
    # TODO: check length of above 3 arrays
    rewards = []  # actual rewards for all bakers in a specific cycle
    rews = cur.execute(
        'select total_income from income_table where cycle = %s and address IN %s' % (cycle, initial_bakers)).fetchall()
    for init_rew in rews:
        rewards.append(init_rew[0])
    n = len(rewards)
    # total_rewards at cycle x for all bakers
    total_reward = cur.execute('select sum(total_income) from income_table where cycle = %s' % cycle).fetchall()[0][0]
    # Fractions: array of length n, where each entry divides rewards[i]/total_rewards[i] for i = 1 to n
    fractions = []
    expected = []
    for c in range(0, n):  # for all the bakers compute the fractions
        if total_reward == 0:
            frac_c = 0
        else:
            frac_c = rewards[c] / total_reward
        fractions.append(frac_c)
        if initial_totals[c] == 0:
            exp_c = 0
        else:
            exp_c = initial_rewards[c] / initial_totals[c]
        expected.append(exp_c)
    return fractions, expected


def plot_expectational_fairness_all_bakers_per_cycle(start, end):
    # TODO: for a cycle look at the exp. fairness for all bakers (1 dotline in plot per cycle)
    x_data = list(range(start, end))  # show cycle start
    y_datas = compute_fraction_all_bakers_one_cycle(start)
    for y_data in y_datas:  # for all cycles a point for every baker in each cycle
        y_data_len = len(y_data)
        plt.plot(x_data * y_data_len, y_data, '.')  # a point for every baker in that cycle
    plt.xlabel('Cycle')
    plt.ylabel('Absolute reward/Expected reward')
    plt.title('Expectational Fairness for cycle' + str(start) + ' and all bakers')
    plt.savefig('images/expectational_fairness/expectational_fairness_allbakers_cycle_' + str(start) + '.png')
    plt.close()


def plot_expectational_fairness_all_bakers_cycles_highest_x_percent(start, end, x):
    x_data = list((range(start, end)))
    y_data_highest, y_data_lowest = compute_fairness_highest_x_percent_all_cycles(start, end, x)
    plt.plot(x_data, y_data_highest, '.', color='black', label='highest ' + str(x) + 'percent')
    plt.plot(x_data, y_data_lowest, '.', color='blue', label='lowest ' + str(x) + 'percent')
    plt.xlabel('Cycle')
    plt.ylabel('Absolute reward/Expected reward')
    plt.legend()
    plt.title('Expectational Fairness from cycle ' + str(start) + ' to ' + str(end) + ' highest ' + str(x) + ' percent')
    plt.savefig('images/expectational_fairness/expectational_fairness_cycles_' + str(start) + '_' + str(
        end) + '_highest_' + str(x) + '.png')
    plt.close()


if __name__ == '__main__':
    # Setup db
    con = sqlite3.connect(DB_FILE)  # attributes: cycle, baker, fee, reward, deposit, blocks (merged db)
    cur = con.cursor()
    baker = "tz3RDC3Jdn4j15J7bBHZd29EUee9gVB1CxD9"
    baker_2 = 'tz3NExpXn9aPNZPorRE4SdjJ2RGrfbJgMAaV'
    start_cycles = [0, 161, 208, 271, 326]
    end_cycles = [160, 207, 270, 325, 397]

    # exp fairness for a single baker
    plot_expectational_fairness(0, 8, baker)
    for start, end in zip(start_cycles, end_cycles):
        plot_expectational_fairness(start, end, baker)

    # expectational fairness with difference of expected and actual reward difference on y axis
    plot_expectational_fairness_difference(0, 8, baker)
    plot_expectational_fairness_difference(0, 8, baker_2)
    for start, end in zip(start_cycles, end_cycles):
        plot_expectational_fairness_difference(start, end, baker)


    # highest x percent
    plot_expectational_fairness_all_bakers_cycles_highest_x_percent(0, 8, 0.1)
    plot_expectational_fairness_all_bakers_cycles_highest_x_percent(0, 398, 0.1)

    #Overview
    plot_expecational_fairness_all_bakers_overview(0, 8, 0.1, 0.25, 0.3)
    plot_expecational_fairness_all_bakers_overview(0, 398, 0.1, 0.25, 0.3)

