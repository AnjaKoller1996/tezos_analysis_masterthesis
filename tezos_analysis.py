import sqlite3
import numpy as np
import matplotlib.pyplot as plt

DB_FILE = '/home/anjakoller/tezos_dataextraction_merged_alltables.db'


def get_avg_reward_over_cycles():
    # get average reward from blocks over all the cycles
    return cur.execute('SELECT AVG(reward) from cyclerewards').fetchone()[0]


# sum up the rewards in a cycle (all the blocks with same cycle) per baker
def get_avg_reward_per_cycle():
    avg_reward_per_cycle = []
    reward_list_tuple = cur.execute('SELECT reward FROM cyclerewards').fetchall()
    for reward in reward_list_tuple:
        avg_reward_per_cycle.append(reward[0])
    return avg_reward_per_cycle


def get_baker_addresses():
    baker_addresses = cur.execute('SELECT DISTINCT(address) FROM bakers').fetchall()
    baker_address_list = []
    for add in baker_addresses:
        baker_address_list.append(add[0])
    return baker_address_list


def get_income_bakers():
    baker_addresses = cur.execute('SELECT DISTINCT(address) FROM income_table').fetchall()
    baker_address_list = []
    for add in baker_addresses:
        baker_address_list.append(add[0])
    return baker_address_list


def get_baker_addresses_from_income_table():
    baker_addresses = cur.execute('SELECT DISTINCT(address) FROM income_table').fetchall()
    baker_address_list = []
    for add in baker_addresses:
        baker_address_list.append(add[0])
    return baker_address_list


def get_baker_addresses_from_block():
    baker_addresses = cur.execute('SELECT DISTINCT(baker) FROM blocks').fetchall()
    baker_address_list = []
    for add in baker_addresses:
        baker_address_list.append(add[0])
    return baker_address_list


def calculate_gini_index(wealths):
    """Compute Gini coefficient of array of values"""
    # convert list to numpy array
    wealths = np.asarray(wealths)
    if np.sum(wealths) == 0:
        return 1
    else:
        diffsum = 0
        for i, xi in enumerate(wealths[:-1], 1):
            diffsum += np.sum(np.abs(xi - wealths[i:]))
        return diffsum / (len(wealths) ** 2 * np.mean(wealths))


def compute_gini_all_bakers_per_cycle(start, end):
    gini_indexes_list = []
    for cycle in range(start, end + 1):
        rewards = []
        rew = cur.execute('SELECT reward FROM blocks WHERE cycle = %s' % cycle).fetchall()
        for r in rew:
            rewards.append(r[0])
        np.asarray(rewards)
        gini = calculate_gini_index(rewards)
        gini_indexes_list.append(gini)
    return gini_indexes_list


def compute_num_working_bakers_per_cycle_list():
    num_bakers_list = []
    num_working_bakers = cur.execute('SELECT working_bakers FROM cycles GROUP BY cycle').fetchall()
    for working_baker in num_working_bakers:
        num_bakers_list.append(working_baker[0])
    print(num_bakers_list)
    return num_bakers_list


def compute_num_active_bakers_per_cycle_list():
    num_bakers_list = []
    num_active_bakers = cur.execute('SELECT active_bakers FROM cycles GROUP BY cycle').fetchall()
    for active_baker in num_active_bakers:
        num_bakers_list.append(active_baker[0])
    print(num_bakers_list)
    return num_bakers_list


def compute_gini_income_table_rolls(start, end):
    """ Gini index of snapshot rolls over all bakers per cycle (start y=cycle <=end)"""
    # for every cycle we have num_baker_addresses entries for the rolls -> compute gini index of them
    gini_indexes_list = []
    for cycle in range(start, end + 1):
        rolls = []
        roll = cur.execute('SELECT rolls FROM income_table WHERE cycle= %s' % cycle).fetchall()
        for r in roll:
            rolls.append(r[0])
        np.asarray(rolls)
        gini_cycle = calculate_gini_index(rolls)
        gini_indexes_list.append(gini_cycle)
    return gini_indexes_list


def compute_gini_income_table_rewards(start, end):
    """ Gini index of snapshot rolls over all bakers per cycle (start y=cycle <=end)"""
    # for every cycle we have num_baker_addresses entries for the total_income -> compute gini index of them
    gini_indexes_list = []
    for cycle in range(start, end + 1):
        incomes = []
        income = cur.execute('SELECT total_income FROM income_table WHERE cycle= %s' % cycle).fetchall()
        for r in income:
            incomes.append(r[0])
        np.asarray(incomes)
        gini_cycle = calculate_gini_index(incomes)
        gini_indexes_list.append(gini_cycle)
    return gini_indexes_list


def compute_reward_era_baker_per_cycle(start, end):
    avg_reward_per_cycle = []
    reward_list_tuple = cur.execute('SELECT reward FROM cyclerewards where cycle>=? AND cycle<=?',
                                    (start, end)).fetchall()
    for reward in reward_list_tuple:
        avg_reward_per_cycle.append(reward[0])
    return avg_reward_per_cycle


# plot the rewards standard deviations for the bakers for all cycles
def plot_reward_standard_deviation_all_cycles():
    avg_reward_per_cycle_list = get_avg_reward_per_cycle()
    avg_reward_over_cycles = get_avg_reward_over_cycles()
    avg_reward_over_cycles_list = [avg_reward_over_cycles] * num_cycles
    x1_data = cycles  # cycle 1 to 397
    y1_data = avg_reward_over_cycles_list  # avg rewards ordered by cycle
    plt.plot(x1_data, y1_data, label='Baseline')
    # show a line where major protocol upgrades are
    plt.axvline(160, 0, 1, label='Babylon 2.0', color='red')
    plt.axvline(208, 0, 1, label='Carthage 2.0', color='red')
    plt.axvline(271, 0, 1, label='Delphi', color='red')
    plt.axvline(325, 0, 1, label='Edo', color='red')
    plt.axvline(355, 0, 1, label='Florence', color='red')
    plt.axvline(387, 0, 1, label='Granada', color='red')
    x2_data = cycles
    y2_data = avg_reward_per_cycle_list  # real rewards per cycle (to see how much it is from the avg)
    plt.plot(x2_data, y2_data, label='Real rewards')
    plt.title('Baker Reward deviation from average reward')
    plt.xlabel('Cycle')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig('images/Baker_reward_deviation_from_average.png')
    plt.close()


def plot_histogram_5cycles_baker_rewards():
    num_bakers_above_baseline = cur.execute('SELECT COUNT(DISTINCT baker) from cyclerewards where reward > '
                                            '161938.98').fetchone()[0]
    num_bakers_below_half = cur.execute('SELECT COUNT(DISTINCT baker) from cyclerewards where reward < '
                                        '161938.98/2').fetchone()[0]
    num_bakers_below_quarter = cur.execute('SELECT COUNT(DISTINCT baker) from cyclerewards where reward < '
                                           '161938.98/4').fetchone()[0]
    num_bakers_above_top = cur.execute('SELECT COUNT(DISTINCT baker) from cyclerewards where reward > '
                                       '163000').fetchone()[0]
    x_data = ['below quarter', 'below half', 'above', 'above top']
    y_data = (num_bakers_below_quarter, num_bakers_below_half, num_bakers_above_baseline, num_bakers_above_top)
    index = np.arange(len(x_data))
    bar_width = 0.9
    plt.bar(index, y_data, bar_width, color="green")
    plt.xticks(index, x_data)  # labels get centered
    plt.title('Distribution of reward amounts among bakers')
    plt.xlabel('How much above or below baseline')
    plt.ylabel('Number of Bakers')
    plt.savefig('images/Histogram_5_cycles_baker_rewards.png')
    plt.close()


def plot_gini_indexes_rewards_all_bakers_per_cycle(start, end):
    gini_indexes_all_bakers_rewards = compute_gini_all_bakers_per_cycle(start, end)
    y_data_length = len(gini_indexes_all_bakers_rewards)
    # ensure that x_data and y_data have same length (can be different due to extracting it at different times)
    x_data = list(range(start, end + 1))
    y_data = gini_indexes_all_bakers_rewards
    plt.ylim(0.0, 0.1)  # make it the same scale as the plots for the stakes
    plt.plot(x_data, y_data)
    plt.title('Gini indexes rewards from ' + str(start) + ' to ' + str(end) + ' all bakers per cycle')
    plt.xlabel('Cycles')
    plt.ylabel('Gini index')
    plt.savefig('rewards_gini/Gini_indexes_all_bakers_' + str(start) + 'to' + str(end) + '_rewards_per_cycle.png')
    plt.close()


def plot_era_baker_reward(start, end, era_name):
    """plot rewards per era, i.e. from cycle start to cycle end"""
    rewards_all_bakers_athens = compute_reward_era_baker_per_cycle(start, end)
    x_data = list(range(start, end + 1))
    y_data = rewards_all_bakers_athens
    plt.plot(x_data, y_data)
    plt.title('Reward all bakers per cycle in Upgrade Era ' + era_name)
    plt.xlabel('Cycles (Time)')
    plt.ylabel('Rewards')
    plt.savefig('rewards_baker/Reward_all_bakers_per_cycle_' + era_name + '.png')
    plt.close()
    return


def plot_num_working_bakers_per_cycle():
    y_data = compute_num_working_bakers_per_cycle_list()
    x_data = list(range(0, len(y_data)))
    plt.plot(x_data, y_data)
    plt.title('Total number of working bakers per cycle')
    plt.xlabel('Cycles')
    plt.ylabel('Number of bakers')
    plt.savefig('images/Total_num_working_bakers_per_cycle.png')
    plt.close()


def plot_num_active_bakers_per_cycle():
    y_data = compute_num_active_bakers_per_cycle_list()
    x_data = list(range(0, len(y_data)))
    plt.plot(x_data, y_data)
    plt.title('Total number of active bakers per cycle')
    plt.xlabel('Cycles')
    plt.ylabel('Number of bakers')
    plt.savefig('images/Total_num_active_bakers_per_cycle.png')
    plt.close()


def plot_income_rolls_gini_index(start, end):
    # select * from income_table where cycle = 150 and address='tz3RDC3Jdn4j15J7bBHZd29EUee9gVB1CxD9';
    # select distinct(address) from income_table;
    baker_addresses = get_baker_addresses_from_income_table()
    gini_indexes_income_table_rolls = compute_gini_income_table_rolls(start, end)
    x_data = list(range(start, end + 1))
    y_data = gini_indexes_income_table_rolls
    plt.plot(x_data, y_data)
    plt.title('Gini indexes income_table rolls from cycles ' + str(start) + ' to ' + str(end))
    plt.xlabel('Cycles')
    plt.ylabel('Gini index')
    plt.savefig('images/income_table_rolls_cycle_' + str(start) + '_to_' + str(end) + '_gini_index.png')
    plt.close()


def plot_income_rewards_gini_index(start, end):
    gini_indexes_income_table_rewards = compute_gini_income_table_rewards(start, end)
    x_data = list(range(start, end + 1))
    y_data = gini_indexes_income_table_rewards
    plt.plot(x_data, y_data)
    plt.title('Gini indexes income_table rewards from cycles ' + str(start) + ' to ' + str(end))
    plt.xlabel('Cycles')
    plt.ylabel('Gini index')
    plt.savefig('images/income_table_rewards_cycle_' + str(start) + '_to_' + str(end) + '_gini_index.png')
    plt.close()


def compute_fractions(start, end, baker):
    """Currently works for 2 cycles and 1 specific baker -> make it work for all bakers and all cycles
    current baker: 'tz3RDC3Jdn4j15J7bBHZd29EUee9gVB1CxD9' """
    rewards = []
    # TODO: try this also out for all addresses -> drop the where address= "x" part
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
    for c in range(start, end+1):
        rew_c = cur.execute('select sum(total_income) from income_table where cycle = %s' % c).fetchall()[0][0]
        total_rewards.append(rew_c)

    # Fractions: array of length n, where each entry divides rewards[i]/total_rewards[i] for i = 1 to n
    fractions = []
    for c in range(0, n):
        frac_c = rewards[c]/total_rewards[c]
        fractions.append(frac_c)
    expected = [initial_reward/initial_total] * n
    return fractions, expected


def plot_expectational_fairness_all_baker_all_cycles():
    """TODO: """


def plot_expectational_fairness(start, end, baker):
    # TODO: make this work for multiple multiple bakers
    """the expectation of the fraction of the reward that baker A receives of the total reward should be equal to his
    initial resource a --> on x axis we have the number of blocks/cycles and on the y axis the fraction of the total
    reward, and another line x_a for the initial resource, start: startcycle/startblock, end: endcycle/endblock
    currently used address: address = 'tz3RDC3Jdn4j15J7bBHZd29EUee9gVB1CxD9'"""
    x_data = list((range(start, end + 1)))
    y_data, y2_data = compute_fractions(start, end, baker)
    plt.plot(x_data, y_data, label='Fraction of resource actual')
    plt.plot(x_data, y2_data, label='Fraction of initial resource (expected)')
    plt.legend()
    plt.xlabel('Cycle')
    plt.ylabel('Fraction of reward')
    plt.title('Expectational Fairness')
    plt.savefig('images/expectational_fairness_' + str(start) + '_' + str(end) + '.png')
    plt.close()


def compute_difference(actual, expected):
    """Computes absolute difference array of the actual and expected values"""
    differences = []
    n = len(actual)
    for c in range(0, n):
        diff_c = np.abs(actual[c]-expected[c])
        differences.append(diff_c)
    return differences


def plot_expectational_fairness_difference(start, end, baker):
    x_data = list((range(start, end + 1)))
    actual, expected = compute_fractions(start, end, baker)
    y_data = compute_difference(actual, expected)
    # y_data is absolute difference of expected and actual value
    plt.plot(x_data, y_data)
    plt.xlabel('Cycle')
    plt.ylabel('Absolute difference of actual and expected reward fraction')
    plt.title('Expectational Fairness')
    plt.savefig('images/expectational_fairness_difference_' + str(start) + '_' + str(end) + '_baker_' + baker +'.png')
    plt.close()


def plot_robust_fairness(address, cycle):
    """Robust fairness Pr((1-epsilon)*a <= gamma_a <= (1+epsilon)*a) <= (1-delta)
    Fix delta, find largest epsilon EPS for which above equation is true
    gamma_a: fraction that baker A receives of total reward,
    a: initial resource of Baker A
    address: address of baker A
    epsilon: between 0 and 1
    delta: >=1
    Version 1 cycle all bakers"""

    EPS = np.empty([100])
    Deltas = np.linspace(0.2, 1, 100)  # take deltas which are higher than 0.18 as there the fluctuations are lower
    Epsilons = np.linspace(0, 1, 100)

    # initial_rewards a
    initial_rewards = []     # list with initial income for all the bakers
    initial_reward = cur.execute('select total_income from income_table where cycle = 0').fetchall()
    for c in range(0, len(initial_reward)):
        rew_c = cur.execute('select total_income from income_table where cycle = 0').fetchall()[c][0]
        initial_rewards.append(rew_c)
    initial_total = cur.execute('select sum(total_income) from income_table where cycle = 0').fetchall()[0][0]
    initial_stakes = []
    for i in initial_rewards:
        init_stake = i/initial_total
        initial_stakes.append(init_stake)

    # fractions gamma_a
    rewards_array = []
    rewards = cur.execute('select total_income from income_table where cycle=%s' % cycle).fetchall()
    for f in range(0, len(rewards)):
        fraction_f = cur.execute('select total_income from income_table where cycle=%s' % cycle).fetchall()[f][0]
        rewards_array.append(fraction_f)

    fractions = []
    total_reward = cur.execute('select sum(total_income) from income_table where cycle=%s' % cycle).fetchall()[0][0]
    for r in rewards_array:
        fraction_r = r/total_reward
        fractions.append(fraction_r)

    # convert initial_rewards and fractions to float array
    initial_stakes = np.fromiter(initial_stakes, dtype=float)
    fractions = np.fromiter(fractions, dtype=float)

    for idx, delta in enumerate(Deltas):
        for eps in Epsilons:
            low_eps = (1-eps)*initial_stakes <= fractions
            high_eps = fractions <= (1+eps)*initial_stakes
            Freq = low_eps * high_eps
            Pr = sum(Freq/len(fractions))
            if Pr >= 1-delta:
                EPS[idx] = eps
                break
    print('EPS', EPS)
    print("The bakers put " + str(round(sum(initial_stakes), 4) * 100) + "% of the stakes and received " + str(
        round(sum(fractions), 4) * 100) + "% of the rewards")
    plt.plot(Deltas, EPS)
    plt.title('Robust Fairness with fixed delta cycle ' + str(cycle))
    plt.xlabel('Delta')
    plt.ylabel('Epsilon')
    plt.savefig('images/robust_fairness_delta_bigger02_cycle_' + str(cycle) + '.png')


if __name__ == '__main__':
    # Setup db
    con = sqlite3.connect(DB_FILE)  # attributes: cycle, baker, fee, reward, deposit, blocks (merged db)
    cur = con.cursor()

    # Variables
    max_cycle = cur.execute('SELECT MAX(cycle) FROM blocks').fetchone()[0]
    num_cycles = max_cycle + 1
    min_cycle = cur.execute('SELECT MIN(cycle) FROM blocks').fetchone()[0]
    cycles = list(range(min_cycle, (max_cycle + 1)))
    num_rolls_total = cur.execute('SELECT SUM(rolls) from snapshots').fetchall()[0]
    num_rolls_self_delegate = \
        cur.execute('SELECT SUM(rolls) from snapshots where address = delegate_addres').fetchall()[0]
    print('Total number of rolls: ', num_rolls_total)
    print('Total number of rolls self-delegate: ', num_rolls_self_delegate)

    # Method/ Plot calls
    plot_reward_standard_deviation_all_cycles()
    plot_histogram_5cycles_baker_rewards()
    # Plot gini indexes of baker rewards for each era

    start_cycles = [0, 161, 208, 271, 326]
    end_cycles = [160, 207, 270, 325, 397]
    for start, end in zip(start_cycles, end_cycles):
        plot_gini_indexes_rewards_all_bakers_per_cycle(start, end)

    # Baker rewards per era
    # cycle_names = ['Athens', 'Babylon', 'Carthage', 'Delphi', 'Edo']
    # for start, end, cycle_name in zip(start_cycles, end_cycles, cycle_names):
    #     plot_era_baker_reward(start, end, cycle_name)

    # get number of active and working bakers per cycle
    plot_num_working_bakers_per_cycle()
    plot_num_active_bakers_per_cycle()

    # Gini index individual eras
    start_cycles = [0, 161, 208, 271, 326]
    end_cycles = [160, 207, 270, 325, 397]
    for start, end in zip(start_cycles, end_cycles):
        plot_income_rolls_gini_index(start, end)
        plot_income_rewards_gini_index(start, end)
    # whole spectrum
    plot_income_rolls_gini_index(0, 390)
    plot_income_rewards_gini_index(0, 390)

    # Expectational Fairness
    baker = "tz3RDC3Jdn4j15J7bBHZd29EUee9gVB1CxD9"
    # TODO: make plot expectational fairness first for only 1 specific baker -> make it work for all bakers
    plot_expectational_fairness(0, 396, baker)
    for start, end in zip(start_cycles, end_cycles):
        plot_expectational_fairness(start, end, baker)

    # expectational fairness with absolute difference of expected and actual reward difference on y axis
    plot_expectational_fairness_difference(0, 396, baker)
    for start, end in zip(start_cycles, end_cycles):
        plot_expectational_fairness_difference(start, end, baker)

    baker_2 = 'tz3NExpXn9aPNZPorRE4SdjJ2RGrfbJgMAaV'
    plot_expectational_fairness_difference(0, 396, baker_2)

    # TODO: do the same for rolls? Assumed that the "stake" is the initial reward that the bakers have at cycle 0

    # Robust fairness (we fix delta and a specific cycle and find epsilon)
    plot_robust_fairness(baker, 1)  # we look at cycle 1 as there we have the same bakers as in cycle 0
    plot_robust_fairness(baker, 5)
    # plot_robust_fairness(baker, 150) # cycle 150 # TODO: take only the bakers that are always active -> not possible to compare as more bakers are there
    # plot_robust_fairness(baker, 200)

    # Close connection
    con.close()
