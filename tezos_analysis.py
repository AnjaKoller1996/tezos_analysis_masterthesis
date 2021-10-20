import sqlite3
import numpy as np
import matplotlib.pyplot as plt


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
    baker_addresses = cur.execute('SELECT address FROM bakers').fetchall()
    baker_address_list = []
    for add in baker_addresses:
        baker_address_list.append(add[0])
    return baker_address_list


def calculate_gini_index(wealths):
    """Compute Gini coefficient of array of values"""
    # convert list to numpy array
    wealths = np.asarray(wealths)
    diffsum = 0
    for i, xi in enumerate(wealths[:-1], 1):
        diffsum += np.sum(np.abs(xi - wealths[i:]))
    return diffsum / (len(wealths) ** 2 * np.mean(wealths))


# TODO: check this
def compute_gini_all_bakers_per_cycle(start, end):
    gini_indexes_list = []
    rewards = []
    baker_address_list = get_baker_addresses()
    # TODO=: check What we want: for each baker (i.e. about 400 baker -> 400 arrays) an array of rewards -> for each
    #  of these we compute the gini and for each cycle we take all the rewards of the bakers and take the gini index
    #  per baker per cycle
    for baker in baker_address_list:
        rew = cur.execute('SELECT reward FROM blocks where baker=? AND cycle >=? AND cycle <=? GROUP BY '
                          'cycle, baker', (baker, start, end)).fetchall()
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
    plt.show()
    plt.savefig('Baker_reward_deviation_from_average.png')
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
    plt.savefig('Histogram_5_cycles_baker_rewards.png')
    plt.show()
    plt.close()


# TODO: needed?
def plot_gini_indexes_rewards_all_bakers_per_cycle(start, end):
    gini_indexes_all_bakers_rewards = compute_gini_all_bakers_per_cycle(start, end)
    y_data_length = len(gini_indexes_all_bakers_rewards)
    # ensure that x_data and y_data have same length (can be different due to extracting it at different times)
    x_data = list(range(0, y_data_length))
    y_data = gini_indexes_all_bakers_rewards
    plt.ylim(0.0, 0.5)  # make it the same scale as the plots for the stakes
    plt.plot(x_data, y_data)
    plt.title('Gini indexes rewards from ' + str(start) + 'to ' + str(end) + ' all bakers per cycle')
    plt.xlabel('Cycles')
    plt.ylabel('Gini index')
    plt.savefig('Gini_indexes_all_bakers_' + str(start) + 'to' + str(end) + '_rewards_per_cycle.png')
    plt.show()
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
    plt.savefig('Reward_all_bakers_per_cycle_' + era_name + '.png')
    plt.show()
    plt.close()
    return


def plot_num_working_bakers_per_cycle():
    y_data = compute_num_working_bakers_per_cycle_list()
    x_data = list(range(0, len(y_data)))
    plt.plot(x_data, y_data)
    plt.title('Total number of working bakers per cycle')
    plt.xlabel('Cycles')
    plt.ylabel('Number of bakers')
    plt.savefig('Total_num_working_bakers_per_cycle.png')
    plt.show()
    plt.close()


def plot_num_active_bakers_per_cycle():
    y_data = compute_num_active_bakers_per_cycle_list()
    x_data = list(range(0, len(y_data)))
    plt.plot(x_data, y_data)
    plt.title('Total number of active bakers per cycle')
    plt.xlabel('Cycles')
    plt.ylabel('Number of bakers')
    plt.savefig('Total_num_active_bakers_per_cycle.png')
    plt.show()
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
    plt.savefig('Snapshot_rolls_cycle' + str(start) + 'to' + str(end) + '_gini_index.png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    # Setup db
    db_file = '/home/anjakoller/tezos_dataextraction_merged_alltables.db'
    con = sqlite3.connect(db_file)  # attributes: cycle, baker, fee, reward, deposit, blocks (merged db)
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
    # TODO: check this does not seem to give a reasonable gini for the rewards
    #plot_gini_indexes_rewards_all_bakers_per_cycle(0, 160)  # takes a while, to debug the rest uncomment this
    #plot_gini_indexes_rewards_all_bakers_per_cycle(161, 207)
    #plot_gini_indexes_rewards_all_bakers_per_cycle(208, 270)
    #plot_gini_indexes_rewards_all_bakers_per_cycle(271, 325)
    #plot_gini_indexes_rewards_all_bakers_per_cycle(326, 397)

    # Snapshot rolls gini index plots
    plot_snapshots_rolls_gini_index(0, 160)  # athens
    plot_snapshots_rolls_gini_index(161, 207)  # babylon
    plot_snapshots_rolls_gini_index(208, 270)  # carthage
    plot_snapshots_rolls_gini_index(271, 325)  # delphi
    plot_snapshots_rolls_gini_index(326, 397)  # edo

    # Baker rewards per era
    # plot_era_baker_reward(0, 160, 'Athens')
    # plot_era_baker_reward(161, 207, 'Babylon')
    # plot_era_baker_reward(208, 270, 'Carthage')
    # plot_era_baker_reward(271, 325, 'Delphi')
    # plot_era_baker_reward(326, 397, 'Edo')

    # get number of active and working bakers per cycle
    plot_num_working_bakers_per_cycle()
    plot_num_active_bakers_per_cycle()

    # Close connection
    cur.close()