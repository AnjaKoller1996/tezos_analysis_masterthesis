import sqlite3
import numpy as np
from scipy.integrate import simps
from numpy import trapz
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


def get_avg_reward_cycle_income_table(start, end):
    avg_reward_list = []
    for c in range(start, end):
        avg_reward_c = cur.execute('SELECT AVG(total_income) from income_table where cycle=%s' % c).fetchall()[0]
        avg_reward_list.append(avg_reward_c)
    return avg_reward_list


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


def get_num_bakers_income_table_per_cycle_list(start, end):
    num_bakers_list = []
    for c in range(start, end):
        num_bakers_c = \
            cur.execute('SELECT COUNT(DISTINCT(address)) FROM income_table where cycle=%s' % c).fetchall()[0][0]
        num_bakers_list.append(num_bakers_c)
    return num_bakers_list


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


def get_cycle_total_rewards():
    """returns a list of the total_rewards for each cycle"""
    total_rewards = []
    for c in range(0, 398):  # iterate over all cycles
        total_rews = cur.execute('SELECT SUM(total_income) FROM income_table where cycle=%s' % c).fetchall()[0][0]
        total_rewards.append(total_rews)
    return total_rewards


def get_active_baker_per_cycle():
    """for each cycle returns a list of active bakers"""
    bakers = []
    for c in range(0, 398):
        baker_c = get_baker_at_cycle(c)
        bakers.append(baker_c)
    return bakers


def get_all_bakers():
    all_bakers = []
    bakers = cur.execute('select distinct(address) from income_table').fetchall()
    for b in bakers:
        all_bakers.append(b[0])
    return all_bakers


def get_baker_first_cycle_init_rew():
    # for all bakers get their first cycle they occur and their reward at this time (init reward)
    bakers = get_all_bakers()
    first_cycles = []
    initial_rewards = []

    for baker in bakers:
        baker_init_rew_cycle = \
            cur.execute('SELECT total_income from income_table where address="%s" limit 1' % baker).fetchall()[0][0]
        initial_rewards.append(baker_init_rew_cycle)
        baker_first_cycle = \
            cur.execute('SELECT cycle from income_table where address="%s" limit 1' % baker).fetchall()[0][0]
        first_cycles.append(baker_first_cycle)
    return bakers, first_cycles, initial_rewards


def get_bakers_initial_values_fairness():
    """return a dicts of bakers initial rewards, rewards, active bakers, first cycles all cycles"""
    total_rewards_all_cycles = get_cycle_total_rewards()
    cycles = [c for c in range(0, 398)]
    cycle_total_reward_dict = dict(zip(cycles, total_rewards_all_cycles))

    bakers_per_cycle = get_active_baker_per_cycle()  # list of lists (for all the 398 cycles a list of bakers)
    cycle_list_of_active_bakers_dict = dict(zip(cycles, bakers_per_cycle))

    # this below has len(bakers) = 381, above arrays have len(cycles = 398
    bakers, first_cycles, initial_rewards = get_baker_first_cycle_init_rew()
    baker_initial_cycle_dict = dict(zip(bakers, first_cycles))
    baker_initial_reward_dict = dict(zip(bakers, initial_rewards))

    return cycle_total_reward_dict, baker_initial_cycle_dict, baker_initial_reward_dict, cycle_list_of_active_bakers_dict


def exp_fairness_one_cycle(cycle, cycle_total_reward_dict,
                           baker_initial_cycle_dict,
                           baker_initial_reward_dict,
                           cycle_list_of_active_bakers_dict):
    """expectational fairness over all bakers just one cycle"""
    rewards_per_baker_in_cycle = dict(
        cur.execute('SELECT address,total_income from income_table where cycle=%s' % cycle).fetchall())
    total_reward_cycle = cycle_total_reward_dict[cycle]
    exp_fairness_baker = []
    for baker in cycle_list_of_active_bakers_dict[cycle]:
        baker_cycle_reward = rewards_per_baker_in_cycle[baker]

        baker_initial_reward = baker_initial_reward_dict[baker]
        baker_initial_cycle = baker_initial_cycle_dict[baker]
        baker_initial_cycle_total_reward = cycle_total_reward_dict[baker_initial_cycle]

        # expectational fairness computation
        if baker_initial_cycle_total_reward == 0 or baker_cycle_reward == 0:
            exp_fairness = 0  # TODO: set this here to 0 or to 1?
        else:
            exp_fairness = (baker_initial_reward / baker_initial_cycle_total_reward) / (
                    baker_cycle_reward / total_reward_cycle)
        exp_fairness_baker.append(exp_fairness)
    return exp_fairness_baker


def compute_expectational_fairness_all_cycles(cycle_total_reward_dict,
                                              baker_initial_cycle_dict,
                                              baker_initial_reward_dict,
                                              cycle_list_of_active_bakers_dict
                                              ):
    """ for all cycles and all bakers which occur in this cycle compute exp. fairness"""
    exp_fairness_list = []
    for cycle in range(0, 398):  # for all cycles
        exp_fairness_baker = exp_fairness_one_cycle(cycle, cycle_total_reward_dict,
                                                    baker_initial_cycle_dict,
                                                    baker_initial_reward_dict,
                                                    cycle_list_of_active_bakers_dict)
        exp_fairness_list.append(exp_fairness_baker)
    return exp_fairness_list  # array of arrays -> for each cycle an array for each baker


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
    gini_indexes_income_table_rolls = compute_gini_income_table_rolls(start, end)
    x_data = list(range(start, end + 1))
    y_data = gini_indexes_income_table_rolls
    plt.plot(x_data, y_data)
    plt.title('Gini indexes income_table rolls from cycles ' + str(start) + ' to ' + str(end))
    plt.xlabel('Cycles')
    plt.ylabel('Gini index')
    plt.savefig('images/gini_index/income_table_rolls_cycle_' + str(start) + '_to_' + str(end) + '_gini_index.png')
    plt.close()


def plot_income_rewards_gini_index(start, end):
    gini_indexes_income_table_rewards = compute_gini_income_table_rewards(start, end)
    x_data = list(range(start, end + 1))
    y_data = gini_indexes_income_table_rewards
    plt.plot(x_data, y_data)
    plt.title('Gini indexes income_table rewards from cycles ' + str(start) + ' to ' + str(end))
    plt.xlabel('Cycles')
    plt.ylabel('Gini index')
    plt.savefig('images/gini_index/income_table_rewards_cycle_' + str(start) + '_to_' + str(end) + '_gini_index.png')
    plt.close()


def plot_expectational_fairness_all_bakers_all_cycles_average(cycle_total_reward_dict, baker_initial_cycle_dict,
                                                              baker_initial_reward_dict,
                                                              cycle_list_of_active_bakers_dict):
    """we have a list for each cycle where every baker has its exp fairness value -> take average per cycle"""
    x_data = list(range(0, 398))
    y_data = compute_expectational_fairness_avg(cycle_total_reward_dict, baker_initial_cycle_dict,
                                                baker_initial_reward_dict,
                                                cycle_list_of_active_bakers_dict)
    plt.plot(x_data, y_data, color='red')
    plt.plot()
    plt.xlabel('Cycle')
    plt.ylabel('Absolute reward/Expected reward')
    plt.title('Expectational Fairness for all cycles and all bakers average')
    plt.savefig('images/expectational_fairness/exp_fairness_allcycles_avg.png')
    plt.close()


def get_exp_fairness_highest_x_percent_bakers(x, cycle_total_reward_dict, baker_initial_cycle_dict, baker_initial_reward_dict, cycle_list_of_active_bakers_dict):
    """returns exp fairness list of the x bakers with the highest values, and then make a histogram"""
    exp_fair_highest_x_list = []
    bakers_highest_x_list = []
    for cycle in range(0, 398):
        exp_fairness_list_c = exp_fairness_one_cycle(cycle, cycle_total_reward_dict, baker_initial_cycle_dict, baker_initial_reward_dict, cycle_list_of_active_bakers_dict)
        bakers = get_baker_at_cycle(cycle)
        # compute how many are the highest x %
        num_entries_per_cycle = len(bakers)
        num_x = int(np.ceil(x/100 * num_entries_per_cycle))
        exp_fairness_dict_c = dict(zip(bakers, exp_fairness_list_c))
        sorted_exp_fairness_dict_c = dict(sorted(exp_fairness_dict_c.items(), key=lambda item: item[1])) # sorted from lowest to highest
        exp_fairness_list_highest_x = list(sorted_exp_fairness_dict_c.items())[-num_x:] # dict with highest x entries (exp fairness and address)
        exp_fair_highest_x_list.append(list(dict(exp_fairness_list_highest_x).values()))
        bakers_highest_x_list.append(list(dict(exp_fairness_list_highest_x).keys()))
    return exp_fair_highest_x_list, bakers_highest_x_list  # 2 array of arrays (one array for each cycle)


def plot_num_occurrences_highest_x_bakers(x, cycle_total_reward_dict, baker_initial_cycle_dict, baker_initial_reward_dict, cycle_list_of_active_bakers_dict):
    """Histogram with occurrences of the highest 5 % of the bakers in each cycle -> how much does which address occur
    (per cycle), once, twice, 5 times,... """
    """on x axis number of occurrences, on y axis how many baker have these num occurrences"""
    exp_fairness_highest_x_list, bakers_highest_x_list = get_exp_fairness_highest_x_percent_bakers(x, cycle_total_reward_dict, baker_initial_cycle_dict, baker_initial_reward_dict, cycle_list_of_active_bakers_dict)
    num_occurrences = []
    bakers = []
    num_bakers_with_occ = []
    bakers_list_flatten = [item for sublist in bakers_highest_x_list for item in sublist]
    for baker in bakers_list_flatten:
        # check if this bakers occurrence is not already tracked, if yes ignore and go to next baker
        if not baker in bakers:
            occ_baker = bakers_list_flatten.count(baker)
            num_occurrences.append(occ_baker) # num_occurrences: how many times each baker in bakers occurs
            bakers.append(baker)  # save the corresponding baker

    occ_contained = []
    for occ in num_occurrences:
        if not occ in occ_contained:  # remove duplicates
            num_bakers_with_occ.append(num_occurrences.count(occ)) # counts how many times a certain occurrencenumber of a baker occurs
            occ_contained.append(occ)
            # TODO: also get the corresponding bakers in order to see which bakers are the ones with the high occurrences
    # np.max(num_occurrences) -> 339 --> num_occurrences.index(339)=24 --> bakers[24]='tz1P7wwnURM4iccy9sSS6Bo2tay9JpuPHzf5'
    # TODO: observe the above baker and some others as well
    x_data = occ_contained
    y_data = num_bakers_with_occ
    index = np.arange(len(x_data))
    bar_width = 0.9
    plt.bar(index, y_data, bar_width, color="green")
    plt.title('Number of occurrences of highest' + str(x) + '% bakers exp. Fairness')
    plt.xlabel('Number of occurrences in highest' + str(x) + '% exp. Fairness')
    plt.ylabel('Number of bakers')
    plt.savefig('images/expectational_fairness/highest_' + str(5) + '_occurrences.png')


def plot_expectational_fairness_highest_x(x,cycle_total_reward_dict,baker_initial_cycle_dict, baker_initial_reward_dict, cycle_list_of_active_bakers_dict):
    """plot the exp fairness of the highest 5 percent of the bakers"""
    exp_fairness_highest_x_list, bakers_highest_x_list = get_exp_fairness_highest_x_percent_bakers(x,
                                                                                                   cycle_total_reward_dict,
                                                                                                   baker_initial_cycle_dict,
                                                                                                   baker_initial_reward_dict,
                                                                                                   cycle_list_of_active_bakers_dict)

    y_datas = exp_fairness_highest_x_list
    x_datas = list(range(0, 398))
    for c in x_datas:  # for every cycle
        x_data = c
        for y_data in y_datas[c]:  # plot all exp fairness values in that cycle of all bakers
            plt.plot(x_data, y_data, '.')
    plt.xlabel('Cycle')
    plt.ylabel('Absolute reward/Expected reward')
    plt.title('Expectational Fairness for all cycles highest ' + str(x) + ' percent bakers')
    plt.savefig('images/expectational_fairness/exp_fairness_highest_' + str(x) + 'bakers.png')
    plt.close()


def plot_expectational_fairness_onecycle(cycle, cycle_total_reward_dict, baker_initial_cycle_dict,
                                         baker_initial_reward_dict, cycle_list_of_active_bakers_dict):
    """plots expectational fairness for 1 specific cycle (all bakers)"""
    y_data = exp_fairness_one_cycle(cycle, cycle_total_reward_dict, baker_initial_cycle_dict, baker_initial_reward_dict,
                                    cycle_list_of_active_bakers_dict)
    x_data = [cycle] * len(y_data)
    plt.plot(x_data, y_data, '.')
    plt.xlabel('Cycle')
    plt.ylabel('Absolute reward/Expected reward')
    plt.title('Expectational Fairness Cycle' + str(cycle))
    plt.savefig('images/expectational_fairness/exp_fairness_cycle_' + str(cycle) + '.png')
    plt.close()


def plot_expectational_fairness_all_bakers_all_cycles(expectational_fairness_list):
    """Plot expectational fairness  for all bakers and cycles"""
    x_datas = list(range(0, 398))
    y_datas = expectational_fairness_list
    for c in x_datas:  # for every cycle
        x_data = c
        for y_data in y_datas[c]:  # plot all exp fairness values in that cycle of all bakers
            plt.plot(x_data, y_data, '.')
    # TODO: for better visibility maybe filter out all that are 0 at first sight
    plt.xlabel('Cycle')
    plt.ylabel('Absolute reward/Expected reward')
    plt.title('Expectational Fairness for all cycles and all bakers (New Version)')
    plt.savefig('images/expectational_fairness/exp_fairness_allbakers_allcycles.png')
    plt.close()


def compute_expectational_fairness_x_quantile(x, cycle_total_reward_dict, baker_initial_cycle_dict,
                                              baker_initial_reward_dict, cycle_list_of_active_bakers_dict):
    """0.05 quantile means 0.05 % smaller, 95% bigger"""
    quantile_list = []
    for c in range(0, 398):
        # list of exp fairness all bakers one cycle -> list
        exp_fairness_c = exp_fairness_one_cycle(c, cycle_total_reward_dict, baker_initial_cycle_dict,
                                                baker_initial_reward_dict,
                                                cycle_list_of_active_bakers_dict)  # this should be an array
        exp_fairness_c_quantile_x = np.quantile(exp_fairness_c, x)
        quantile_list.append(exp_fairness_c_quantile_x)
    return quantile_list


def plot_expectational_fairness_x_quantile(x, cycle_total_reward_dict, baker_initial_cycle_dict,
                                           baker_initial_reward_dict, cycle_list_of_active_bakers_dict):
    y_data = compute_expectational_fairness_x_quantile(x, cycle_total_reward_dict, baker_initial_cycle_dict,
                                                       baker_initial_reward_dict, cycle_list_of_active_bakers_dict)
    x_data = list(range(0, 398))
    plt.plot(x_data, y_data)
    plt.title('Expectational Fairness ' + str(x) + ' quantile')
    plt.xlabel('Cycles')
    plt.ylabel('Absolute reward/Expected reward')
    plt.savefig('images/expectational_fairness/exp_fairness_quantile_' + str(x) + '.png')
    plt.close()


def plot_exp_fairness_overview(cycle_total_reward_dict, baker_initial_cycle_dict,
                               baker_initial_reward_dict, cycle_list_of_active_bakers_dict):
    """avg, 5% quantile, 95% quantile and median in one plot"""
    x_data = list(range(0, 398))
    y1_data = compute_expectational_fairness_avg(cycle_total_reward_dict, baker_initial_cycle_dict,
                                                 baker_initial_reward_dict,
                                                 cycle_list_of_active_bakers_dict)
    plt.plot(x_data, y1_data, label='Avg')
    y2_data = compute_expectational_fairness_x_quantile(0.05, cycle_total_reward_dict, baker_initial_cycle_dict,
                                                        baker_initial_reward_dict, cycle_list_of_active_bakers_dict)
    plt.plot(x_data, y2_data, label='5% quantile')
    y3_data = compute_expectational_fairness_x_quantile(0.95, cycle_total_reward_dict, baker_initial_cycle_dict,
                                                        baker_initial_reward_dict, cycle_list_of_active_bakers_dict)
    plt.plot(x_data, y3_data, label='95% quantile')
    y4_data = compute_expectational_fairness_x_quantile(0.5, cycle_total_reward_dict, baker_initial_cycle_dict,
                                                        baker_initial_reward_dict, cycle_list_of_active_bakers_dict)
    plt.plot(x_data, y4_data, label='Median')
    plt.title('Expectational Fairness Overview')
    plt.legend()
    plt.xlabel('Cycles')
    plt.ylabel('Absolute reward/Expected reward')
    plt.savefig('images/expectational_fairness/exp_fairness_overview.png')
    plt.close()


def compute_expectational_fairness_avg(cycle_total_reward_dict, baker_initial_cycle_dict, baker_initial_reward_dict,
                                       cycle_list_of_active_bakers_dict):
    """Computes avg exp fairness in each cycle"""
    exp_fairness_avg_list = []
    for c in range(0, 398):
        # list of exp fairness all bakers one cycle -> list
        exp_fairness_c = exp_fairness_one_cycle(c, cycle_total_reward_dict, baker_initial_cycle_dict,
                                                baker_initial_reward_dict,
                                                cycle_list_of_active_bakers_dict)  # this should be an array
        exp_fairness_c_avg = np.mean(exp_fairness_c)
        exp_fairness_avg_list.append(exp_fairness_c_avg)
    return exp_fairness_avg_list


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


def get_stakes_and_fractions(cycle):
    """Computes robust fairness for a specific cycle for all bakers"""
    cycle_total_reward_dict, baker_initial_cycle_dict, baker_initial_reward_dict, cycle_list_of_active_bakers_dict = get_bakers_initial_values_fairness()

    initial_stakes = []
    fractions = []

    robust_fairness_list = []
    rewards_per_baker_in_cycle = dict(
        cur.execute('SELECT address,total_income from income_table where cycle=%s' % cycle).fetchall())
    total_reward_cycle = cycle_total_reward_dict[cycle]
    for baker in cycle_list_of_active_bakers_dict[cycle]:
        baker_cycle_reward = rewards_per_baker_in_cycle[baker]
        baker_initial_reward = baker_initial_reward_dict[baker]
        baker_initial_cycle = baker_initial_cycle_dict[baker]
        baker_initial_cycle_total_reward = cycle_total_reward_dict[baker_initial_cycle]

        if baker_initial_cycle_total_reward == 0 or baker_cycle_reward == 0:
            init_stake = 0
            fraction = 0
        else:
            init_stake = baker_initial_reward / baker_initial_cycle_total_reward
            fraction = baker_cycle_reward / total_reward_cycle
        initial_stakes.append(init_stake)
        fractions.append(fraction)

    return initial_stakes, fractions


def get_bakers_initial_rewards_at_cycle(cycle):
    # initialize initial bakers and rewards array
    bakers = get_baker_at_cycle(0)
    rewards = get_income_at_cycle(0)
    initial_totals = [cur.execute('SELECT SUM(total_income) from income_table where cycle=0').fetchall()[0][0]] * 8
    current_bakers = get_baker_at_cycle(cycle)
    # for every cycle from 0 to cycle, check if baker already exist, if not add it and add its initial_reward
    for c in range(0, cycle + 1):
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

    # Remove the bakers that occur from cylces 0 to current but are not active in current
    bakers_to_remove = list(set(current_bakers).symmetric_difference(set(bakers)))
    remove_indexes = []
    if not bakers_to_remove == []:
        for b in bakers_to_remove:
            index = bakers.index(b)
            remove_indexes.append(index)

        for index in remove_indexes:
            del (bakers[index])
            del (initial_totals[index])
            del (rewards[index])
    return bakers, rewards, initial_totals


def compute_robust_fairness_old(cycle, Deltas=np.linspace(0, 1, 100)):
    initial_bakers, initial_rewards, initial_totals = get_bakers_initial_rewards_at_cycle(cycle)
    EPS = np.empty([100])
    # Deltas = np.linspace(0, 1, 100)
    Epsilons = np.linspace(0, 50000, 100)

    initial_stakes = []  # check if this array has correct length
    n = len(initial_rewards)
    for i in range(0, n):
        if initial_totals[i] == 0:
            init_stake = 0
        else:
            init_stake = initial_rewards[i] / initial_totals[i]
        initial_stakes.append(init_stake)

    # fractions gamma_a
    rewards_array = get_income_at_cycle(cycle)  # length of num_bakers at cycle (for ex at cycle 5 this has length 8)
    fractions = []  # reward at cycle x divided by total_reward at cycle_x, same length as rewards_array
    total_reward = cur.execute('select sum(total_income) from income_table where cycle=%s' % cycle).fetchall()[0][0]
    for r in rewards_array:
        if total_reward == 0:
            fraction_r = 0
        else:
            fraction_r = r / total_reward
        fractions.append(fraction_r)

    # convert initial_rewards and fractions to float array
    initial_stakes = np.fromiter(initial_stakes, dtype=float)  # length of num bakers at cycle 0
    fractions = np.fromiter(fractions, dtype=float)  # length of num bakers at cycle (i.e. current cycle)

    for idx, delta in enumerate(Deltas):
        for eps in Epsilons:
            low_eps = (1 - eps) * initial_stakes <= fractions
            high_eps = fractions <= (1 + eps) * initial_stakes
            Freq = low_eps * high_eps
            Pr = sum(Freq) / len(fractions)
            if Pr >= 1 - delta:
                EPS[idx] = eps
                break
    # print('EPS', EPS)
    # print("The bakers put " + str(round(sum(initial_stakes), 4) * 100) + "% of the stakes and received " + str(
    #    round(sum(fractions), 4) * 100) + "% of the rewards")
    return Deltas, EPS


# old implementation -> this way we need to high EPS to satisfy equation
def compute_robust_fairness(cycle, Deltas=np.linspace(0, 1, 100)):
    """Robust fairness for one specific cycle and all bakers"""
    # Note: for some cycles like 130 the EPS value would need to be immensly high to satisfy this
    EPS = np.linspace(0, 1, 100)
    Epsilons = np.linspace(0, 50000, 100)

    initial_stakes, fractions = get_stakes_and_fractions(cycle)
    initial_stakes = np.fromiter(initial_stakes, dtype=float)  # length of num bakers at cycle 0
    fractions = np.fromiter(fractions, dtype=float)  # length of num bakers at cycle (i.e. current cycle)
    for idx, delta in enumerate(Deltas):
        for eps in Epsilons:
            low_eps = (1 - eps) * initial_stakes <= fractions
            high_eps = fractions <= (1 + eps) * initial_stakes
            Freq = low_eps * high_eps
            Pr = sum(Freq) / len(fractions)
            if Pr >= 1 - delta:
                EPS[idx] = eps
                break
    return Deltas, EPS


def plot_robust_fairness_old(cycle):
    x_data, y_data = compute_robust_fairness_old(cycle)
    plt.plot(x_data, y_data)
    plt.title('Robust Fairness with fixed delta cycle ' + str(cycle))
    plt.xlabel('Delta')
    plt.ylabel('Epsilon')
    plt.savefig('images/robust_fairness/robust_fairness_cycle_' + str(cycle) + '.png')
    plt.close()


def plot_robust_fairness(cycle):
    x_data, y_data = compute_robust_fairness(cycle)
    plt.plot(x_data, y_data)
    plt.title('Robust Fairness with fixed delta cycle ' + str(cycle))
    plt.xlabel('Delta')
    plt.ylabel('Epsilon')
    plt.savefig('images/robust_fairness/robust_fairness_old_cycle_' + str(cycle) + '.png')
    plt.close()


def get_num_baker_per_cycle(start, end):
    """returns the number of bakers in each cycle from cycle start to end"""
    num_bakers_per_cycle = []  # total number of bakers in each cycle
    for cycle in range(start, end):
        nb_per_cycle = cur.execute('SELECT COUNT() from income_table where cycle=%s' % cycle).fetchall()[0][0]
        num_bakers_per_cycle.append(nb_per_cycle)
    return num_bakers_per_cycle


def get_baker_rewards_per_cycle_sorted(cycle):
    """returns an array of baker rewards for each cycle sorted from highest reward to lowest reward"""
    baker_rewards_sorted = []  # one entry for every baker
    baker_rewards = cur.execute('SELECT total_income from income_table where cycle=%s' % cycle).fetchall()
    for br in baker_rewards:
        baker_rewards_sorted.append(br[0])
    baker_rewards_sorted.sort(reverse=True)  # sorts descending
    return baker_rewards_sorted


def compute_nakamoto_index(start, end):
    num_cycles = end - start
    num_bakers = [0] * num_cycles  # array with number of bakers per cycle needed for > 50%
    total_rewards_per_cycle = get_total_rewards_per_cycle(start, end)
    num_bakers_per_cycle = get_num_baker_per_cycle(start, end)

    for c in range(start, end):  # for every cycle
        baker_rewards_sorted_c = get_baker_rewards_per_cycle_sorted(c)
        baker_rewards_c_summed = 0
        num_bakers_c = 0
        i = 0
        while i <= len(baker_rewards_sorted_c):  # while not all bakers in the cycle watched
            if baker_rewards_c_summed > (total_rewards_per_cycle[c] / 2):  # reach >50%
                num_bakers[c] = num_bakers_c
                break  # stop while loop when this condition is met
            else:  # if not > 50% then add one (current baker), and add the reward of the current baker
                num_bakers_c += 1
                baker_rewards_c_summed += baker_rewards_sorted_c[i]
                i += 1

    num_bakers_fraction = []
    for n in num_bakers:
        num_bakers_fraction.append(n / num_bakers_per_cycle[n])
    return num_bakers_fraction


def compute_aoc(cycle):
    """computes the area under the curve for robust fairness for all the cycles from start to end"""
    Deltas, EPS = compute_robust_fairness(cycle)
    EPS = np.asarray(EPS)
    areas = np.sum(0.01 * EPS)
    # other solution trapezoidal rule=> areas = trapz(EPS, dx=0.01)
    return areas


def plot_nakamoto_index(start, end):
    """plots the nakamoto index for each cycle, i.e. the relative number of bakers (percentage) we need in order to
    have more than 50% of the stake, this can fluctuate, but if it has a general tendency to go down, then it gets
    unfair as we have a few very big players """
    x_data = list((range(start, end)))
    y_data = compute_nakamoto_index(start, end)
    plt.plot(x_data, y_data)
    plt.xlabel('Cycle')
    plt.ylabel('Percentage of bakers to reach > 50%')
    plt.title('Nakamoto Index for each cycle from ' + str(start) + ' to ' + str(end))
    plt.savefig('images/Nakamoto_index' + str(start) + '_' + str(end) + '.png')
    plt.close()


def plot_nakamoto_index_num_bakers(start, end):
    """Plots nakamoto index on y axis and num_bakers (in each cycle) on x axis"""
    x_data = get_num_bakers_income_table_per_cycle_list(start, end)  # get number of bakers in each cycle
    y_data = compute_nakamoto_index(start, end)
    plt.plot(x_data, y_data, '.')
    plt.xlabel('Number of Bakers (Cycle)')
    plt.ylabel('Percentage of bakers to reach > 50%')
    plt.title('Nakamoto Index for each cycle from ' + str(start) + ' to ' + str(end))
    plt.savefig('images/Nakamoto_index_num_bakers' + str(start) + '_' + str(end) + '.png')
    plt.close()


def compute_mean_rewards(start, end):
    """Computes mean reward per cycle from cycle start to end"""
    mean_rewards = []
    mean_reward = cur.execute(
        'SELECT AVG(total_income) from income_table where cycle >=%s and cycle <=%s group by cycle' % (
            start, end)).fetchall()
    for mr in mean_reward:
        mean_rewards.append(mr[0])
    return mean_rewards


def compute_median_rewards(start, end):
    """Computed median reward per cycle from cycle start to end"""
    rewards = get_rewards_all_bakers(start, end)
    median_rewards = []
    for x in range(0, len(rewards)):  # iterate over all bakers, take median in each cycle
        rewards_c = rewards[x]
        rewards_c_sorted = np.sort(rewards_c)
        median_c_index = int(np.ceil(len(rewards_c) / 2))
        median_c = rewards_c_sorted[median_c_index]
        median_rewards.append(median_c)
    return median_rewards


def get_total_rewards_per_cycle(start, end):
    """returns total rewards for each cycle from start to end"""
    total_rewards = []
    total_reward = cur.execute('SELECT SUM(total_income) from income_table where cycle >=%s and cycle <=%s group by '
                               'cycle' % (start, end)).fetchall()
    for tr in total_reward:
        total_rewards.append(tr[0])
    return total_rewards


def get_rewards_all_bakers(start, end):
    """rewards is an array of length cycles (end-start) and for each entry at c there is an array with length
    num_bakers """
    rewards = []
    for c in range(start, end + 1):  # for every cycle
        rewards_c = get_income_at_cycle(c)
        rewards.append(rewards_c)
    return rewards


def get_relative_rewards(start, end):
    """returns the relative rewards (relative to the total) for every cycle and baker"""
    total_rewards = get_total_rewards_per_cycle(start, end)
    rewards = get_rewards_all_bakers(start, end)
    relative_rewards = []
    for c in range(start, end + 1):  # for every cycle
        rewards_relative_c = [item / total_rewards[c] for item in rewards[c]]
        relative_rewards.append(rewards_relative_c)
    return relative_rewards


def get_relative_rewards_mean(start, end):
    total_rewards = get_total_rewards_per_cycle(start, end)
    mean_rewards = compute_mean_rewards(start, end)
    mean_rewards_relative = []
    for c in range(start, end + 1):
        mean_rewards_relative_c = mean_rewards[c] / total_rewards[c]
        mean_rewards_relative.append(mean_rewards_relative_c)
    return mean_rewards_relative


def plot_distance_to_mean(start, end):
    """Plots the relative difference the bakers rewards have compared to the mean total reward per cycle averaged
    over all bakers per cycle"""
    x_data = list(range(start, end + 1))
    y2_data = get_relative_rewards_mean(start, end)  # array of length cycle
    y_data = get_relative_rewards(start, end)  # array (len cycles) of arrays (len num_bakers)
    plt.plot(x_data, y2_data, color='red', label='relative mean')
    for c in x_data:
        for baker in y_data[c]:
            plt.plot(c, baker, '.')  # for every baker in the cycle make a '.'
    plt.legend()
    plt.xlabel('Cycle')
    plt.ylabel('Relative distance of total_reward to mean')
    plt.title('Relative Distance to Mean per Cycle from ' + str(start) + ' to ' + str(end))
    plt.savefig('images/income_distance_' + str(start) + '_' + str(end) + '.png')
    plt.close()


def plot_robust_fairness_aoc(start, end):
    for x in range(start, end):
        y_data = compute_aoc(x)
        plt.plot(x, y_data, '.')
    plt.xlabel('Cycle')
    plt.ylabel('Area')
    plt.title('Robust fairness Area under Curve per Cycle from ' + str(start) + ' to ' + str(end))
    plt.savefig('images/robust_fairness/area_under_curve_' + str(start) + '_' + str(end) + '.png')
    plt.close()


def plot_median_and_avg_reward(start, end):
    x_data = list(range(start, end + 1))
    y1_data = compute_median_rewards(start, end)
    y2_data = compute_mean_rewards(start, end)
    plt.plot(x_data, y1_data, label='Median')
    plt.plot(x_data, y2_data, label='Mean')
    plt.legend()
    plt.xlabel('Cycle')
    plt.ylabel('Rewards')
    plt.title('Mean and Avg rewards per Cycle from ' + str(start) + ' to ' + str(end))
    plt.savefig('images/mean_avg_rewards_' + str(start) + '_' + str(end) + '.png')
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

    # Compute median and avg reward for all cycles in a plot
    plot_median_and_avg_reward(0, 6)
    plot_median_and_avg_reward(0, 398)

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
    plot_num_working_bakers_per_cycle()  # TODO: take this bakers from the income_Table
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
    # TODO: make an exp. fairness plot for 1 specific baker and a difference plot

    cycle_total_reward_dict, baker_initial_cycle_dict, baker_initial_reward_dict, cycle_list_of_active_bakers_dict = get_bakers_initial_values_fairness()

    exp_fairness_list = compute_expectational_fairness_all_cycles(cycle_total_reward_dict, baker_initial_cycle_dict,
                                                                  baker_initial_reward_dict,
                                                                  cycle_list_of_active_bakers_dict)

    exp_fairness_highest5, highest5_bakers = get_exp_fairness_highest_x_percent_bakers(5, cycle_total_reward_dict, baker_initial_cycle_dict,
                                                                  baker_initial_reward_dict,
                                                                  cycle_list_of_active_bakers_dict)

    plot_num_occurrences_highest_x_bakers(5, cycle_total_reward_dict, baker_initial_cycle_dict,
                                                                  baker_initial_reward_dict,
                                                                  cycle_list_of_active_bakers_dict)

    plot_expectational_fairness_highest_x(5, cycle_total_reward_dict, baker_initial_cycle_dict,
                                              baker_initial_reward_dict, cycle_list_of_active_bakers_dict)

    # Exp. Fairness overview plot (avg, lowest 5% resp. 1%, highest 5% resp. 1% in one plot)
    plot_exp_fairness_overview(cycle_total_reward_dict, baker_initial_cycle_dict,
                               baker_initial_reward_dict,
                               cycle_list_of_active_bakers_dict)
    plot_expectational_fairness_x_quantile(0.05, cycle_total_reward_dict, baker_initial_cycle_dict,
                                           baker_initial_reward_dict, cycle_list_of_active_bakers_dict)
    plot_expectational_fairness_x_quantile(0.95, cycle_total_reward_dict, baker_initial_cycle_dict,
                                           baker_initial_reward_dict, cycle_list_of_active_bakers_dict)
    plot_expectational_fairness_all_bakers_all_cycles(exp_fairness_list)
    plot_expectational_fairness_all_bakers_all_cycles_average(cycle_total_reward_dict, baker_initial_cycle_dict,
                                                              baker_initial_reward_dict,
                                                              cycle_list_of_active_bakers_dict)  # average over all bakers in each cycle

    # expectational fairness at individual cycles for all bakers, look at cycles in each era
    # TODO: note if there are differences among the eras, compare several cycles in different eras
    era_cycles = [50, 130, 200, 250, 300, 340, 370, 395]
    for cycle in era_cycles:
        plot_expectational_fairness_onecycle(cycle, cycle_total_reward_dict, baker_initial_cycle_dict,
                                             baker_initial_reward_dict, cycle_list_of_active_bakers_dict)

    # Robust fairness (we fix delta and a specific cycle and find epsilon) Note: for robust fairness cycles higher
    # than 6 we have very high EPS values and therefore plots look wrong at first sight
    #plot_robust_fairness_old(6)
    #plot_robust_fairness_old(50)
    plot_robust_fairness(50)
    plot_robust_fairness(6)
    # plot a robust fairness for every era (one cycle in each era)
    for c in era_cycles:
        plot_robust_fairness(c)

    # Area under curve robust fairness
    # TODO: comment this out as it takes long
    # plot_robust_fairness_aoc(0, 5)  # works for every cycle (for initial value we take the value in prev. cycle)
    # plot_robust_fairness_aoc(0, 398)

    # Compute mean reward plot, i.e. cycles on x axis, percentage of total reward on y axis, plot a line with the mean
    # reward over all bakers in red and the relative rewards for all bakers per cycle on the y axis
    # TODO: eventually or detect the bakers that have a too big distance
    plot_distance_to_mean(0, 8)
    plot_distance_to_mean(0, 398)

    #  Nakamoto index
    plot_nakamoto_index(0, 398)
    plot_nakamoto_index(0, 8)

    # Nakamoto index with num bakers in network on x axis
    plot_nakamoto_index_num_bakers(0, 8)
    plot_nakamoto_index_num_bakers(0, 398)

    # Close connection
    con.close()
