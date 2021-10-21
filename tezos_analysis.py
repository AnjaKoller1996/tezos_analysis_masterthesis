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


def compute_proportion_snapshot_rolls(start, end):
    # TODO: corresponds to the fairness plot
    """ Computed the proportion of rolls in a snapshot compared to all the rolls (for each baker) """
    proportion_rolls_list = []
    return proportion_rolls_list


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


def compute_gini_snapshot_rolls(start, end):
    """Note: a roll denotes 8000tz"""
    """Gini index of snapshot rolls over all bakers per cycle (start <= cycle <= end)"""
    gini_indexes_list = []
    for cycle in range(start, end + 1):
        rolls = []
        roll = cur.execute('SELECT rolls FROM snapshots WHERE cycle = %s' % cycle).fetchall()
        for r in roll:
            rolls.append(r[0])
        np.asarray(rolls)
        gini_cycle = calculate_gini_index(rolls)
        gini_indexes_list.append(gini_cycle)
    return gini_indexes_list


def compute_proportion_rewards_all_bakers_per_cycle(start, end):
    """Computes a list for each baker with the proportion of the rolls compared to the total rolls in the respective
    cycles """
    # TODO: implement this, corresponds to fairness plot
    proportion_rolls_list = []
    return proportion_rolls_list


def compute_reward_era_baker_per_cycle(start, end):
    avg_reward_per_cycle = []
    reward_list_tuple = cur.execute('SELECT reward FROM cyclerewards where cycle>=? AND cycle<=?',
                                    (start, end)).fetchall()
    for reward in reward_list_tuple:
        avg_reward_per_cycle.append(reward[0])
    return avg_reward_per_cycle


def compute_base_reward_propotion():
    # TODO: implement the avg/base proportion expected fairness
    return 1


def compute_base_rolls_proportion():
    # TODO: implement avg/base proportion expected fairness
    return 1


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


# TODO: How much did a baker earn per cycle, snapshot t-7 corresponds to cycle t --> rewards per baker per cycle
# then compare with snapshot t-7
def plot_snapshots_rolls_gini_index(start, end):
    # take only some cycle's snapshots to look at individual sections
    gini_indexes_all_bakers_snapshot = compute_gini_snapshot_rolls(start, end)
    x_data = list(range(start, end + 1))
    y_data = gini_indexes_all_bakers_snapshot
    plt.plot(x_data, y_data)
    plt.title('Gini indexes Snapshot rolls (reward) from cycles ' + str(start) + ' to ' + str(end))
    plt.xlabel('Snapshots Cycles')
    plt.ylabel('Gini index')
    plt.ylim(0.6, 0.95)  # make it the same scale as the plots for rewards
    plt.savefig('snapshots/Snapshot_rolls_cycle_' + str(start) + '_to_' + str(end) + '_gini_index.png')
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

    # Snapshot rolls gini index plots
    # TODO: check this does not seem to give a reasonable gini for the rewards -> DEBUG -> TOO many snapshots
    for start, end in zip(start_cycles, end_cycles):
        plot_snapshots_rolls_gini_index(start, end)

    # TODO: fairness measure -> in a cycle compare proportion of rolls and proportion of rewards -> define an "ok
    #  band" and a stochastic variation which is ok
    plot_fairness_comparison_rewards_rolls(250, 251, 0.03)  # look at cycle 250 and have a variation of 3% tolerance

    # Baker rewards per era
    # cycle_names = ['Athens', 'Babylon', 'Carthage', 'Delphi', 'Edo']
    # for start, end, cycle_name in zip(start_cycles, end_cycles, cycle_names):
    #     plot_era_baker_reward(start, end, cycle_name)

    # get number of active and working bakers per cycle
    plot_num_working_bakers_per_cycle()
    plot_num_active_bakers_per_cycle()

    # Close connection
    con.close()

# Note: The right to publish a block in cycle n is assigned to a randomly selected roll in a randomly selected
# roll snapshot from cycle n - PRESERVED_CYCLES — 2 (which is n-7)
# roll n-7 must be somehow coherent with rewards n (rewards are payed out 7 cycles after snapshot is taken)
