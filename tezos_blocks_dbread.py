import sqlite3
import numpy as np
import matplotlib.pyplot as plt


def get_avg_reward_over_cycles():
    # get average reward from blocks over all the cycles
    return cur.execute('SELECT AVG(reward) from cyclerewards').fetchone()[0]


def calculate_gini_index(wealths):
    """Compute Gini coefficient of array of values"""
    # convert list to numpy array
    wealths = np.asarray(wealths)
    diffsum = 0
    for i, xi in enumerate(wealths[:-1], 1):
        diffsum += np.sum(np.abs(xi - wealths[i:]))
    return diffsum / (len(wealths) ** 2 * np.mean(wealths))


# sum up the rewards in a cycle (all the blocks with same cycle) per baker
def get_avg_reward_per_cycle():
    avg_reward_per_cycle = []
    reward_list_tuple = cur.execute('SELECT reward FROM cyclerewards').fetchall()
    for reward in reward_list_tuple:
        avg_reward_per_cycle.append(reward[0])
    return avg_reward_per_cycle


def get_avg_reward_blocks():
    avg_reward_blocks = []
    reward_list_tuple = cur.execute('SELECT reward from blocks').fetchall()
    for reward in reward_list_tuple:
        avg_reward_blocks.append(reward[0])
    return avg_reward_blocks


# plot the rewards standard deviations for the bakers for all cycles
def plot_reward_standarddeviation_all_cycles():
    avg_reward_per_cycle_list = get_avg_reward_per_cycle()
    avg_reward_over_cycles = get_avg_reward_over_cycles()
    avg_reward_over_cycles_list = [avg_reward_over_cycles] * num_cycles
    x1_data = cycles  # cycle 1 to 397
    y1_data = avg_reward_over_cycles_list  # avg rewards ordered by cycle
    plt.plot(x1_data, y1_data, label='baseline')
    # show a line where major protocol upgrades are
    plt.axvline(160, 0, 1, label='Babylon 2.0', color='red')
    plt.axvline(208, 0, 1, label='Carthage 2.0', color='red')
    plt.axvline(271, 0, 1, label='Delphi', color='red')
    plt.axvline(325, 0, 1, label='Edo', color='red')
    plt.axvline(355, 0, 1, label='Florence', color='red')
    plt.axvline(387, 0, 1, label='Granade', color='red')
    x2_data = cycles
    y2_data = avg_reward_per_cycle_list  # real rewards per cycle (to see how much it is from the avg)
    plt.plot(x2_data, y2_data, label='real rewards')
    plt.title('Baker Reward deviation from average reward')
    plt.xlabel('Cycle')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()
    plt.savefig('Baker_reward_deviation_from_average.png')
    plt.close()


def plot_histogram_5cycles_baker_rewards():
    avg_reward_over_cycles = cur.execute('SELECT AVG(reward) from cyclerewards WHERE cycle IN (250, 251, 252, '
                                         '253, 254 )').fetchone()[0]
    num_bakers_above_baseline = cur.execute('SELECT COUNT(DISTINCT baker) from cyclerewards where reward > '
                                            '161938.98').fetchone()[0]
    num_bakers_below_half = cur.execute('SELECT COUNT(DISTINCT baker) from cyclerewards where reward < '
                                        '161938.98/2').fetchone()[0]
    num_bakers_below_quarter = cur.execute('SELECT COUNT(DISTINCT baker) from cyclerewards where reward < '
                                           '161938.98/4').fetchone()[0]
    num_bakers_above_top = cur.execute('SELECT COUNT(DISTINCT baker) from cyclerewards where reward > '
                                       '163000').fetchone()[0]
    x_data = ['below quarter', 'belowhalf', 'above', 'abovetop']
    y_data = (num_bakers_below_quarter, num_bakers_below_half, num_bakers_above_baseline, num_bakers_above_top)
    index = np.arange(len(x_data))
    bar_width = 0.9
    plt.bar(index, y_data, bar_width, color="green")
    plt.xticks(index, x_data)  # labels get centered
    plt.title('Distribution of reward amounts among bakers')
    plt.xlabel('How much above or below baseline')
    plt.ylabel('Number of Bakers')
    plt.savefig('Histo_5cycles_baker_rewards.png')
    plt.show()
    plt.close()


def compute_gini_index_rewards(rewards):
    reward_list = []
    for reward in rewards:
        reward_list.append(reward[0])
    gini_index_rewards = calculate_gini_index(reward_list)
    return gini_index_rewards


def get_baker_addresses():
    baker_addresses = cur.execute('SELECT address FROM bakers').fetchall()
    baker_address_list = []
    for add in baker_addresses:
        baker_address_list.append(add[0])
    return baker_address_list


def compute_gini_all_bakers_per_cycle():
    gini_indexes_list = []
    rewards = []
    baker_address_list = get_baker_addresses()
    for baker in baker_address_list:
        rew = cur.execute('SELECT reward FROM blocks where baker=? GROUP BY cycle,baker', (baker,)).fetchall()
        for r in rew:
            rewards.append(r[0])
        np.asarray(rewards)
        gini = calculate_gini_index(rewards)
        gini_indexes_list.append(gini)
    return gini_indexes_list


def compute_gini_all_bakers_staking_per_block():
    """TODO: use this and look at it in a block (i.e.not cycle) perspective """
    # SELECT ROW_NUMBER() OVER(ORDER BY (SELECT 1)) AS MyIndex FROM blocks;
    stakes = []
    gini_indexes_list = []
    baker_address_list = get_baker_addresses()
    for baker in baker_address_list:
        # TODO: check this again for correctnes --> we have the blocks with info about cycle, baker, reward,
        #  the cycles which know which baker is associated, and the baker table which has staking info
        stake = cur.execute('SELECT staking_balance FROM baker_staking_cycles WHERE baker=? GROUP BY cycle, baker',
                            (baker,)).fetchall()
        for s in stake:
            stakes.append(s[0])
        np.asarray(stakes)
        gini = calculate_gini_index(stakes)
        gini_indexes_list.append(gini)
    return gini_indexes_list


def compute_gini_all_bakers_staking_per_cycle():
    '''TODO: look at snapshot'''
    gini_indexes_list = []
    stakes = []
    baker_address_list = get_baker_addresses()
    for baker in baker_address_list:
        # TODO: check this again for correctnes --> we have the blocks with info about cycle, baker, reward,
        #  the cycles which know which baker is associated, and the baker table which has staking info
        stake = cur.execute('SELECT staking_balance FROM baker_staking_cycles WHERE baker=? GROUP BY cycle, baker',
                            (baker,)).fetchall()
        for s in stake:
            stakes.append(s[0])
        np.asarray(stakes)
        gini = calculate_gini_index(stakes)
        gini_indexes_list.append(gini)
    return gini_indexes_list


def compute_num_bakers_per_cycle_list():
    num_bakers_list = []
    cycles = list(range(0, num_cycles))
    for cycle in cycles:
        num_baker = cur.execute('SELECT working_bakers from cycles where cycle=?', (cycle,)).fetchall()
        num_bakers_list.append(num_baker[0])
    print(num_bakers_list)
    return num_bakers_list


def compute_num_stakes_per_cycle_list():
    num_stakes_list = []
    cycles = list(range(0, num_cycles))
    for cycle in cycles:
        num_baker = cur.execute('SELECT staking_supply from cycles where cycle=?', (cycle,)).fetchall()
        num_stakes_list.append(num_baker[0])
    print(num_stakes_list)
    return num_stakes_list


def plot_gini_indexes_all_bakers_per_cycle():
    gini_indexes_all_bakers_rewards = compute_gini_all_bakers_per_cycle()
    y_data_length = len(gini_indexes_all_bakers_rewards)
    # ensure that x_data and y_data have same length (can be different due to extracting it at different times)
    x_data = list(range(0, y_data_length))
    y_data = gini_indexes_all_bakers_rewards
    plt.ylim(0.0, 0.5)  # make it the same scale as the plots for the stakes
    plt.plot(x_data, y_data)
    plt.title('Gini indexes all bakers per cycle')
    plt.xlabel('Cycles')
    plt.ylabel('Gini index')
    plt.savefig('Gini_indexes_all_bakers_per_cycle.png')
    plt.show()
    plt.close()


def plot_gini_indexes_all_bakers_staking_balance_per_cycle():
    gini_indexes_all_bakers_staking = compute_gini_all_bakers_staking_per_cycle()
    y_data_length = len(gini_indexes_all_bakers_staking)
    x_data = list(range(0, y_data_length))
    y_data = gini_indexes_all_bakers_staking
    plt.plot(x_data, y_data)
    plt.title('Gini indexes all bakers staking_balance per cycle')
    plt.xlabel('Cycles')
    plt.ylabel('Gini index')
    plt.savefig('Gini_indexes_all_bakers_staking_per_cycle.png')
    plt.show()
    plt.close()


def plot_num_bakers_per_cycle():
    """TODO: integrate the new cycles table take working_bakers instead of the cyclerewards table"""
    x_data = list(range(0, num_cycles))
    y_data = compute_num_bakers_per_cycle_list()
    plt.plot(x_data, y_data)
    plt.title('Total number of bakers per cycle')
    plt.xlabel('Cycles')
    plt.ylabel('Number of bakers')
    plt.savefig('Total_num_bakers_per_cycle.png')
    plt.show()
    plt.close()


def plot_total_amount_of_stakes_per_cycle():
    x_data = list(range(0, num_cycles))
    y_data = compute_num_stakes_per_cycle_list()
    plt.plot(x_data, y_data)
    plt.title('Total number of stakes per cycle')
    plt.xlabel('Cycles')
    plt.ylabel('Number of stakes (stakes percentage)')
    plt.savefig('Total_num_stakes_per_cycle.png')
    plt.show()
    plt.close()


def compute_gini_all_bakers_per_cycle_era(start, end):
    gini_indexes_list = []
    rewards = []
    baker_address_list = get_baker_addresses()
    for baker in baker_address_list:
        rew = cur.execute('SELECT reward FROM blocks where baker=? and cycle >=? and cycle <=? GROUP BY cycle,baker',
                          (baker, start, end)).fetchall()
        for r in rew:
            rewards.append(r[0])
        np.asarray(rewards)
        gini = calculate_gini_index(rewards)
        gini_indexes_list.append(gini)
    return gini_indexes_list


def plot_era_reward_gini(start, end, era_name):
    gini_indexes_all_bakers_rewards = compute_gini_all_bakers_per_cycle_era(start, end)
    y_data = gini_indexes_all_bakers_rewards
    y_data_length = len(y_data)
    x_data = list(range(0, y_data_length))
    plt.ylim(0.0, 0.5)  # make it the same scale as the plots for the stakes
    plt.plot(x_data, y_data)
    plt.title('Era ' + era_name + ' Gini indexes all bakers per cycle')
    plt.xlabel('Cycles')
    plt.ylabel('Gini index')
    plt.savefig('Era' + era_name + '_Gini_indexes_all_bakers_per_cycle.png')
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


def compute_reward_era_baker_per_cycle(start, end):
    avg_reward_per_cycle = []
    reward_list_tuple = cur.execute('SELECT reward FROM cyclerewards where cycle>=? AND cycle<=?',
                                    (start, end)).fetchall()
    for reward in reward_list_tuple:
        avg_reward_per_cycle.append(reward[0])
    return avg_reward_per_cycle


def compute_gini_stakes_era(start, end):
    gini_indexes_list = []
    stakes = []
    baker_address_list = get_baker_addresses()
    for baker in baker_address_list:
        stake = cur.execute('SELECT staking_balance FROM baker_staking_cycles WHERE baker=? and cycle >=? and cycle '
                            '<=? GROUP BY cycle, baker', (baker, start, end)).fetchall()
        for s in stake:
            stakes.append(s[0])
        np.asarray(stakes)
        gini = calculate_gini_index(stakes)
        gini_indexes_list.append(gini)
    return gini_indexes_list


def compute_gini_snapshot_rolls(start, end):
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


def compute_gini_snapshot_rewards():
    """TODO: implement this so that we get gini index list of all baker rewards per snapshot"""
    gini_indexes_list = []
    return gini_indexes_list


def plot_era_stakes_gini(start, end, era_name):
    gini_indexes_all_bakers_staking = compute_gini_stakes_era(start, end)
    y_data_length = len(gini_indexes_all_bakers_staking)
    x_data = list(range(0, y_data_length))
    y_data = gini_indexes_all_bakers_staking
    plt.plot(x_data, y_data)
    plt.title('Era ' + era_name + ' Gini indexes stakes')
    plt.xlabel('Cycles')
    plt.ylabel('Gini index')
    plt.savefig('Era' + era_name + 'Gini_indexes_stakes.png')
    plt.show()
    plt.close()


def plot_snapshots_rolls_gini_index():
    # take only some cycle's snapshots to look at individual sections
    start = 161
    end = 207
    gini_indexes_all_bakers_snapshot = compute_gini_snapshot_rolls(start, end)
    y_data_length = len(gini_indexes_all_bakers_snapshot)
    x_data = list(range(start, end +1))
    y_data = gini_indexes_all_bakers_snapshot
    print('ydata lenght', y_data_length)
    print('y data last entries', y_data[:-5])
    plt.plot(x_data, y_data)
    plt.title('Gini indexes Snapshot rolls from cycles ' + str(start) + ' to ' + str(end))
    plt.xlabel('Snapshots Cycles')
    plt.ylabel('Gini index')
    plt.savefig('Snapshot_rolls_cycle' + str(start) + 'to' + str(end) + '_gini_index.png')
    plt.show()
    plt.close()


def plot_snapshot_rewards_gini_index():
    # TODO: take only some snapshots for testing it, then take all if possible with runtime
    start = 0
    end = 5
    gini_indexes_all_bakers_rewards = compute_gini_snapshot_rewards(start, end)
    y_data_length = len(gini_indexes_all_bakers_rewards)
    x_data = list(range(0, y_data_length))
    y_data = gini_indexes_all_bakers_rewards
    plt.plot(x_data, y_data)
    plt.title('Gini indexes Snapshot rewards')
    plt.xlabel('Snapshots')
    plt.ylabel('Gini index')
    plt.savefig('Snapshot_rewards_gini_index.png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    dbfile = '/home/anjakoller/tezos_dataextraction_merged_alltables.db'
    con = sqlite3.connect(dbfile)  # attributes: cycle, baker, fee, reward, deposit, blocks (merged db)
    cur = con.cursor()
    num_bakers = cur.execute('SELECT COUNT(DISTINCT reward) FROM blocks').fetchone()[0]
    max_cycle = cur.execute('SELECT MAX(cycle) FROM blocks').fetchone()[0]
    num_cycles = max_cycle + 1
    min_cycle = cur.execute('SELECT MIN(cycle) FROM blocks').fetchone()[0]
    cycles = list(range(min_cycle, (max_cycle + 1)))
    num_blocks = cur.execute('SELECT COUNT(*) FROM BLOCKS').fetchone()[0]
    plot_reward_standarddeviation_all_cycles()
    plot_histogram_5cycles_baker_rewards()
    total_rewards_per_bakers = cur.execute('SELECT total_rewards_earned from accounts').fetchall()
    gini_index_totalrew_per_baker = compute_gini_index_rewards(total_rewards_per_bakers)

    # plot_gini_indexes_all_bakers_per_cycle()  # TODO: This takes a while, to debug the rest uncomment this
    plot_gini_indexes_all_bakers_staking_balance_per_cycle()

    plot_era_baker_reward(0, 160, 'Athens')  # cycles 0 to 160 athens
    plot_era_baker_reward(161, 207, 'Babylon')  # cycles 161 to 208 babylon
    plot_era_baker_reward(208, 270, 'Carthage')  # cycles 209 to 271 carthage
    plot_era_baker_reward(271, 325, 'Delphi')  # cycle 271 to 325 delphi
    plot_era_baker_reward(326, 397, 'Edo')  # cycle 325 to today edo

    # all gini indexes 5 eras for upgrades individually
    # TODO: comment them in after as they need some time
    # plot_era_reward_gini(0, 160, 'Athens')  # cycles 0 to 160
    # plot_era_reward_gini(161, 207, 'Babylon')  # cycles 161 to 208 babylon
    # plot_era_reward_gini(208, 270, 'Carthage')  # cycles 209 to 271 carthage
    # plot_era_reward_gini(271, 325, 'Delphi')  # cycle 271 to 325 delphi
    # plot_era_reward_gini(326, 397, 'Edo')  # cycle 325 to today edo

    # all gini indexes staking 5 eras for upgrades individually
    # plot_era_stakes_gini(0, 160, 'Athens')  # cycles 0 to 160 athens stakes
    # plot_era_stakes_gini(161, 207, 'Babylon')  # cycles 161 to 208 babylon
    # plot_era_stakes_gini(208, 270, 'Carthage')  # cycles 209 to 271 carthage
    # plot_era_stakes_gini(271, 325, 'Delphi')  # cycle 271 to 325 delphi
    # plot_era_stakes_gini(326, 397, 'Edo')  # cycle 325 to today edo

    plot_num_bakers_per_cycle()
    plot_total_amount_of_stakes_per_cycle()
    # TODO: look at a snapshot and plot the gini over a section of blocks (i.e. a snapshot)
    plot_snapshots_rolls_gini_index() # can be called for a certain amount of cyclesnapshots
    # plot_snapshot_rewards_gini_index() # TODO: implement
    cur.close()

# TODO: make a plot with x axis staking of bakers (all bakers on x axis), y axis rewards of the bakers
# just do it for a few cycles and select randomly (i.e .for higher stake we should have higher reward -> 45 degree)


# TODO: take snapshots --> get gini indexes for rolls in snapshots and look at a sequence of snapshots TODO: get get
#  distribution of rolls for all bakers in a snapshot and calculate gini index for bakers, associate snapshot data
#  and cycle TODO: get percentage of staking_balance of a baker compared to the staking_supply total that a cycle has
#   TODO: make a plot with on x axis staking of bakers (all bakers on x axis), y axis rewards of the bakers ,
#    just do it for a few cycles and select randomly --> theoretically for higher stake -> higher reward (somehow 45
#    degree slope) (for few cycles only) TODO: total amount of bakers, amount of staking in total


# TODO: important -> control all the array that they sum up in the correct way