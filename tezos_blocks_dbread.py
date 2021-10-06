import sqlite3
import numpy as np
import matplotlib.pyplot as plt


def get_avg_reward_over_cycles():
    # get average reward from blocks over all the cycles
    return cur.execute('SELECT AVG(reward) from cyclerewards').fetchone()[0]


def get_reward_over_5cyclesfromx():
    rewards = []
    # TODO: make this more general working with range x...y
    avg_rewards = cur.execute('SELECT reward from cyclerewards WHERE cycle IN (250, 251, 252, 253, 254 )').fetchall()
    for rew in avg_rewards:
        rewards.append(rew)
    return rewards


def get_reward_per_cycle_baker(cycle_num):
    rewards_list = []
    rewards = cur.execute('SELECT reward FROM blocks where cycle=? GROUP BY cycle,baker', (cycle_num,)).fetchall()[0]
    for re in rewards:
        rewards_list.append(re)
    return rewards_list


def calculate_gini_index(wealths):
    """Compute Gini coefficient of array of values"""
    # convert list to numpy array
    wealths = np.asarray(wealths)
    # print('wealths', wealths)
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
    plt.axvline(208, 0, 1, label='Carthago 2.0', color='red')
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


def plot_reward_distribution_blocks():
    avg_reward_block_list = get_avg_reward_blocks()
    average_reward_over_blocks = cur.execute('SELECT AVG(reward) from blocks').fetchone()[0]
    averaged_rewards_perblock = [average_reward_over_blocks] * num_blocks
    blocks = cur.execute('SELECT COUNT(*) FROM BLOCKS').fetchone()[0]
    x1_data = list(range(0, blocks))
    y1_data = avg_reward_block_list
    plt.title('Baker reward distribution per block')
    plt.plot(x1_data, y1_data, label='real rewards')
    # x2_data = list(range(0,blocks))
    # y2_data = averaged_rewards_perblock
    # plt.plot(x2_data, y2_data, label='baseline rewards')
    plt.xlabel('blocks')
    plt.ylabel('rewards')
    # plt.legend()
    plt.show()
    plt.savefig('Baker_reward_distribution_blocks.png')
    plt.close()


# Detail view where we look at 5 consecutive cycles (nr 250 to 254)
def plot_5cycles_reward_baker():
    reward_per_cycle_list_250to254_list = get_reward_over_5cyclesfromx()
    reward_per_cycle_baker_list = get_reward_per_cycle_baker()
    x1_data = [250, 251, 252, 253, 254]  # the respective cycles we look at
    y1_data = reward_per_cycle_list_250to254_list
    plt.plot(x1_data, y1_data, label='baseline')
    x2_data = [250, 251, 252, 253, 254]
    y2_data = reward_per_cycle_baker_list  # real reward data of a specific baker in all the cycles
    plt.plot(x2_data, y2_data, label='Real rewards for baker X')
    plt.legend()
    plt.title('Baker Reward in Cycles 250 to 255 for Baker X compared to baseline')
    plt.savefig('Baker_reward_cycles_250to255_bakerX.png')
    plt.show()
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


def compute_gini_all_bakers_staking_per_cycle():
    gini_indexes_list = []
    stakes = []
    baker_address_list = get_baker_addresses()
    for baker in baker_address_list:
        stake = cur.execute('SELECT staking_balance FROM baker_staking_cycles WHERE baker=? GROUP BY cycle, baker', (baker,)).fetchall()
        for s in stake:
            stakes.append(s[0])
        np.asarray(stakes)
        gini = calculate_gini_index(stakes)
        gini_indexes_list.append(gini)
    return gini_indexes_list


def compute_num_bakers_per_cycle_list():
    num_bakers_list = []
    cycles = list(range(0,num_cycles))
    for cycle in cycles:
        num_baker = cur.execute('SELECT COUNT(*) from bakers where cycle=?', (cycle,).fetchall())
        num_bakers_list.append(num_baker[0])
    print(num_bakers_list)
    return num_bakers_list


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
    x_data = list(range(0, num_cycles))
    y_data = compute_num_bakers_per_cycle_list()
    plt.plot(x_data, y_data)
    plt.title('Total number of bakers per cycle')
    plt.xlabel('Cycles')
    plt.ylabel('Number of bakers')
    plt.savefig('Total_num_bakers_per_cycle')
    plt.show()
    plt.close()


def plot_era_reward(start, end):
    """plot rewards per era, i.e. from cycle start to cycle end"""
    return


def compute_gini_index_era_per_cycle():
    # TODO:
    return


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
    plot_reward_distribution_blocks()
    # plot_5cycles_reward_baker()
    plot_histogram_5cycles_baker_rewards()
    total_rewards_per_bakers = cur.execute('SELECT total_rewards_earned from accounts').fetchall()
    gini_index_totalrew_per_baker = compute_gini_index_rewards(total_rewards_per_bakers)

    # plot_gini_indexes_all_bakers_per_cycle()  # This takes a while, to debug the rest uncomment this # TODO: run this again and test if the ylim works
    plot_gini_indexes_all_bakers_staking_balance_per_cycle()

    # TODO: put this creation here in a loop and save all 5 plots
    plot_era_reward(0, 160)  # cycles 0 to 160 athens
    plot_era_reward(161, 208)   # cycles 161 to 208 babylon
    plot_era_reward(209, 271)  # cycle 209 to 271 carthage
    plot_era_reward(271, 325)  # cycle 271 to 325 delphi
    plot_era_reward(325, 404)   # cycle 325 to today edo

    # plot_num_bakers_per_cycle() # TODO: make this
    # plot_total_amount_of_stakes_per_cycle() # TODO: make this
    # TODO: look at a snapshot and plot the gini over a section of blocks (i.e. a snapshot)
    cur.close()
