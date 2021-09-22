# import the tezos blocks db and the snapshot db and create a new one
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# creating file path
dbfile = '/home/anjakoller/tezos_dataextraction_merged_alltables.db'

# Create a SQL connection to our SQLite database with blocks
con = sqlite3.connect(dbfile)  # attributes: cycle, baker, fee, reward, deposit, blocks (merged db)
cur = con.cursor()

for row in cur.execute('SELECT * FROM blocks where cycle = 1 LIMIT 10'):
    print(row)

# order the entries per cycle and plot the reward per cycle per baker over time (over the different cycles) -> take
# entries of same cycle together check how many distinct baker addresses there are -> for all cycles add up the reward
# which have the same cycle and same baker address and plot this over time (i.e. over the different cycles)
num_bakers = cur.execute('SELECT COUNT(DISTINCT reward) FROM blocks').fetchone()[0]
max_cycle = cur.execute('SELECT MAX(cycle) FROM blocks').fetchone()[0]
num_cycles = max_cycle + 1
min_cycle = cur.execute('SELECT MIN(cycle) FROM blocks').fetchone()[0]
cycles = list(range(min_cycle, (max_cycle + 1)))


def compute_cyclereward_perbaker_table():
    cur.execute('SELECT cycle, baker, SUM(reward) FROM blocks Group by cycle')
    '''below is the sql code to fill the table cyclerewards
    INSERT INTO cyclereward (reward, cycle, baker)
     VALUES(SELECT SUM(reward),cycle,baker FROM blocks Group by cycle));
     INSERT INTO cyclerewards (reward, cycle, baker)
      SELECT SUM(reward),cycle,baker
      FROM blocks
      GROUP BY cycle;'''


def get_avg_reward_over_cycles():
    # get average reward from blocks over all the cycles
    return cur.execute('SELECT AVG(reward) from cyclerewards').fetchone()[0]


def get_reward_over_5cyclesfromx():
    rewards = []
    # TODO: make this more general working with range x...y
    # avg_rewards = cur.execute('SELECT AVG(reward) from cyclerewards WHERE cycle IN (x, x+1, x+2, x+3,
    # x+4 )').fetchall()
    avg_rewards = cur.execute('SELECT reward from cyclerewards WHERE cycle IN (250, 251, 252, 253, 254 )').fetchall()
    for rew in avg_rewards:
        rewards.append(rew)
    return rewards


def avg_reward_over_cycles_250to254():
    rewards = []
    avg_rewards = cur.execute(
        'SELECT AVG(reward) from cyclerewards WHERE cycle IN (250, 251, 252, 253, 254 )').fetchall()
    for rew in avg_rewards:
        rewards.append(rew)
    return rewards


def get_reward_per_cycle_baker(address):
    # TODO
    rewards = []
    return rewards


def calculate_gini_index(wealths):
    """Compute Gini coefficient of array of values"""
    # convert list to numpy array
    wealths = np.asarray(wealths)
    diffsum = 0
    for i, xi in enumerate(wealths[:-1], 1):
        diffsum += np.sum(np.abs(xi - wealths[i:]))
    return diffsum / (len(wealths)**2 * np.mean(wealths))


def compute_gini_simple(x):
    mean_abs_diff = np.abs(np.subtract.outer(x, x)).mean()
    rel_mean_abs_diff = mean_abs_diff / np.mean(x)
    # Gini coefficient
    g = 0.5 * rel_mean_abs_diff
    return g


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


def get_sum_reward_per_cycle():
    reward_cycle = []
    for cycle in cycles:
        reward_sum = \
            cur.execute('SELECT SUM(reward) FROM (SELECT * FROM blocks b1, blocks b2 WHERE b1.cycle = b2.cycle '
                        'AND b1.baker != b2.baker)').fetchone()[0]
        reward_cycle.append(reward_sum)
    return reward_cycle


avg_reward_over_cycles = get_avg_reward_over_cycles()
avg_reward_over_cycles_list = [avg_reward_over_cycles] * num_cycles
avg_reward_per_cycle_list = get_avg_reward_per_cycle()
avg_reward_block_list = get_avg_reward_blocks()
num_blocks = cur.execute('SELECT COUNT(*) FROM BLOCKS').fetchone()[0]
average_reward_over_blocks = cur.execute('SELECT AVG(reward) from blocks').fetchone()[0]
averaged_rewards_perblock = [average_reward_over_blocks] * num_blocks

print('avg reward over cycles', avg_reward_over_cycles)
print('avg reward per cycle list', avg_reward_per_cycle_list)
print('last 10 entries avg reward per cycle', avg_reward_per_cycle_list[-9:])


# plot the rewards standard deviations for the bakers for all cycles
def plot_reward_standarddeviation_all_cycles():
    x1_data = cycles  # cycle 1 to 397
    print(x1_data)
    y1_data = avg_reward_over_cycles_list  # avg rewards ordered by cycle
    print(y1_data)
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
    blocks = cur.execute('SELECT COUNT(*) FROM BLOCKS').fetchone()[0]
    print('blocks', blocks)
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
    avg_reward_over_cycles_250to254_list = avg_reward_over_cycles_250to254()
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


plot_reward_standarddeviation_all_cycles()

plot_reward_distribution_blocks()
# plot_5cycles_reward_baker() # TODO: execute this when done
avg_reward_over_cycles = cur.execute('SELECT AVG(reward) from cyclerewards WHERE cycle IN (250, 251, 252, '
                                     '253, 254 )').fetchone()[0]
print('avg reward over the 5 cycles', avg_reward_over_cycles)
# TODO: possibly try to get median here
num_bakers_above_baseline = cur.execute('SELECT COUNT(DISTINCT baker) from cyclerewards where reward > '
                                        '161938.98').fetchone()[0]
num_bakers_below_half = cur.execute('SELECT COUNT(DISTINCT baker) from cyclerewards where reward < '
                                    '161938.98/2').fetchone()[0]
num_bakers_below_quarter = cur.execute('SELECT COUNT(DISTINCT baker) from cyclerewards where reward < '
                                       '161938.98/4').fetchone()[0]
num_bakers_above_top = cur.execute('SELECT COUNT(DISTINCT baker) from cyclerewards where reward > '
                                   '163000').fetchone()[0]

print('Number of bakers above the baseline', num_bakers_above_baseline)  # 21
print('num bakers below half', num_bakers_below_half)  # 71
print('num bakers below quarter', num_bakers_below_quarter)  # 103
print('num bakers above 163000', num_bakers_above_top)  # 2


def plot_histogram_5cycles_baker_rewards():
    x_data = ['below quarter', 'belowhalf', 'above', 'abovetop']
    num_bakers_above_baseline = cur.execute('SELECT COUNT(DISTINCT baker) from cyclerewards where reward > '
                                            '161938.98').fetchone()[0]
    y_data = (num_bakers_below_quarter, num_bakers_below_half, num_bakers_above_baseline, num_bakers_above_top)
    index = np.arange(len(x_data))
    bar_width = 0.9
    print(y_data)
    print(index)
    plt.bar(index, y_data, bar_width, color="green")
    plt.xticks(index, x_data)  # labels get centered
    plt.title('Distribution of reward amounts among bakers')
    plt.xlabel('How much above or below baseline')
    plt.ylabel('Number of Bakers')
    plt.show()
    plt.close()


plot_histogram_5cycles_baker_rewards()


def compute_gini_index_rewards(rewards):
    reward_list = []
    for reward in rewards:
        reward_list.append(reward[0])
    gini_index_rewards = calculate_gini_index(reward_list)
    print(gini_index_rewards)
    return gini_index_rewards


bakers_rewards = cur.execute('SELECT reward from cyclerewards').fetchall()
gini_index_allbakers = compute_gini_index_rewards(bakers_rewards)  # Gini index over all cycles for all bakers
print('GINI INDEX each cycle all bakers', gini_index_allbakers)


total_rewards_per_bakers = cur.execute('SELECT total_rewards_earned from accounts').fetchall()
gini_index_totalrew_per_baker = compute_gini_index_rewards(total_rewards_per_bakers)
print('GINI INDEX total reward per baker', gini_index_totalrew_per_baker)


# get the reward per bakers and cycle and look at the gini index for specific bakers over all cycles
def calculate_gini_all_cycles():
    return

# TODO2: Make same as above for the bakers also for the delegators & compare bakers and delegators


# close the connection
cur.close()

# TODO: calculate gini index for each cycle --> see how it develops over time check the take off point at 200 cycles
#  -> check if they see amount of tokens check proposal -> voting and online to check whether proposal/protocol
#  changed at certain amount of cycles like 200, 150 look at self-delegates vs. bakers and nonself-delegates vs.
#  bakers calculate gini index for the bakers staking_balance & plot how to select baker as a delegator -> sleect
#  ones that already have a large stake? -> could be problematic similar to rich get richer--> gini index of stakes
#  gives insight (if this distribution becomes more unequal) called preferential attechment
