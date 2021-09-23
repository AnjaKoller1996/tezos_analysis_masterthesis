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
    rewards = cur.execute('SELECT reward FROM blocks where cycle=? GROUP BY cycle,baker', (cycle_num,)).fetchall()
    for rew in rewards:
        rewards_list.append(rew)
    return rewards_list


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


def compute_gini_all_bakers_per_cycle():
    baker_per_cycle_gini_indexes = []
    rewards_list = []
    cycles = list(range(num_cycles))
    for cycle in cycles:
        rewards_list.append(get_reward_per_cycle_baker(cycle))
    for rew in rewards_list:
        baker_per_cycle_gini_indexes.append(compute_gini_index_rewards(rew))
    return baker_per_cycle_gini_indexes


def plot_gini_indexes_all_bakers_per_cycle():
    gini_indexes_all_bakers_rewards = compute_gini_all_bakers_per_cycle()
    x_data = list(range(0, num_cycles))
    y_data = gini_indexes_all_bakers_rewards
    plt.plot(x_data, y_data)
    plt.title('Gini indexes all bakers per cycle')
    plt.xlabel('Cycles')
    plt.ylabel('Gini index')
    plt.savefig('Gini_indexes_all_bakers_per_cycle')
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
    plot_reward_distribution_blocks()
    # plot_5cycles_reward_baker()
    plot_histogram_5cycles_baker_rewards()
    total_rewards_per_bakers = cur.execute('SELECT total_rewards_earned from accounts').fetchall()
    gini_index_totalrew_per_baker = compute_gini_index_rewards(total_rewards_per_bakers)

    plot_gini_indexes_all_bakers_per_cycle()
    cur.close()

# TODO: Make same as above for the bakers also for the delegators & compare bakers and delegators & do it also for
#  staking calculate gini index for each cycle --> see how it develops over time check the take off point at 200 cycles
#  -> check if they see amount of tokens check proposal -> voting and online to check whether proposal/protocol
#  changed at certain amount of cycles like 200, 150 look at self-delegates vs. bakers and nonself-delegates vs.
#  bakers calculate gini index for the bakers staking_balance & plot how to select baker as a delegator -> sleect
#  ones that already have a large stake? -> could be problematic similar to rich get richer--> gini index of stakes
#  gives insight (if this distribution becomes more unequal) called preferential attechment
