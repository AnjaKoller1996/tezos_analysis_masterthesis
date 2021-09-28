# db changes here

# TODO 1) baker_cycle_staking table
# Baker: address, stakingbalance, Cycle: cycle, activebakers, activedelegators
# insert into cyclerewards staking_balance select staking_balance from bakers where cyclerewards.baker = bakers.address;


# SELECT cyclerewards.baker, cyclerewards.cycle, cyclerewards.reward, bakers.address, bakers.staking_balance,
# bakers.staking_capacity, bakers.id FROM cyclerewards INNER JOIN bakers ON cyclerewards.baker = bakers.address;


# create table baker_staking_cycles AS SELECT cyclerewards.baker, cyclerewards.cycle, cyclerewards.reward,
# bakers.address, bakers.staking_balance, bakers.staking_capacity, bakers.id FROM cyclerewards INNER JOIN bakers ON
# cyclerewards.baker = bakers.address;

# import sqlite3

# dbfile = '/home/anjakoller/tezos_dataextraction_merged_alltables.db'
# dbfile2 = '/home/anjakoller/tezos_'
# con = sqlite3.connect(dbfile)  # attributes: cycle, baker, fee, reward, deposit, blocks (merged db)
# cur = con.cursor()