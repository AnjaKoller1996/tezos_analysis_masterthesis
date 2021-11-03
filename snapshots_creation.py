import sqlite3
import requests
import pprint
import pickle

conn = sqlite3.connect('snapshots_data.db', timeout=10)
c = conn.cursor()  # db saved in location where .py file is save

# Bakers
LOAD_FROM_CACHE = True

baker_response = None
if not LOAD_FROM_CACHE:
    api_url_bakers = 'https://api.tzstats.com/explorer/bakers'
    params = {'limit': 10000}
    r = requests.get(api_url_bakers, params=params)
    baker_response = r.json()
    # pprint(baker_response[0])
    with open("bakers", "wb") as outfile:
        pickle.dump(baker_response, outfile)
else:
    with open("bakers", "rb") as infile:
        baker_response = pickle.load(infile)


snapshot_table_api_url = 'https://api.tzstats.com/tables/snapshot'
block_table_api_url = 'https://api.tzstats.com/tables/block' # cycle=150&is_cycle_snapshot=1
snapshots = []
addresses = [r['address'] for r in baker_response]
cycles = list(range(0, 396)) # array with all the cycles

for cycle in cycles:   # for all bakers:
    # get all snapshots per baker
    params = {'limit': 10000}
    # https: // api.tzstats.com / tables / block?cycle = 150 & is_cycle_snapshot = 1
    r = requests.get(block_table_api_url + '?cycle=' + cycle + "&is_cycle_snapshot=1",
                     params=params)
    snapshot_response = r.json()
    snapshots.extend(snapshot_response)

snapshots_length = len(snapshots)


class BlockTable:
    def __init__(self, row_id, is_orphan, cycle, fitness, baker_id, fee, reward, deposit, baker ):
        self.row_id = row_id
        self.is_orphan = is_orphan
        self.cycle = cycle
        self.fitness = fitness
        self.baker_id = baker_id          # unique row id of the block's baker account
        # TODO: extract the row_id of the baker accounts --> get baker table again with the row_id and rolls information
        # row_id -> block_table[0], is_orphan [3], cycle [5], fitness [11], baker_id


class Snapshot:
    def __init__(self, cycle, account_id, delegate_id, is_delegate, balance, delegated, address, delegate_address):
        self.cycle = cycle
        self.account_id = account_id
        self.delegate_id = delegate_id
        self.is_delegate = is_delegate
        self.balance = balance
        self.delegated = delegated
        self.address = address
        self.delegate_address = delegate_address


# create a snapshot table
c.execute(''' SELECT count(name) from sqlite_master where type='table' AND name='snapshots' ''')

if c.fetchone()[0] == 1:
    c.execute('DROP TABLE snapshots')
    c.execute(
        '''CREATE TABLE snapshots (cycle integer, account_id integer, delegate_id integer, is_delegate boolean, 
        balance integer, delegated boolean, address text, delegate_address text)''')
else:
    c.execute(
        '''CREATE TABLE snapshots (cycle integer, account_id integer, delegate_id integer, is_delegate boolean, 
        balance integer, delegated boolean, address text, delegate_address text)''')


def insert_snapshot(snshot):
    with conn:
        c.execute(
            'INSERT INTO snapshots VALUES(:cycle, :account_id, :delegate_id, :is_delegate, :balance, :delegated, '
            ':address, :delegate_address)',
            {'cycle': snshot.cycle, 'account_id': snshot.account_id, 'delegate_id': snshot.delegate_id,
             'is_delegate': snshot.is_delegate, 'balance': snshot.balance, 'delegated': snshot.delegated,
             'address': snshot.address, 'delegate_address': snshot.delegate_address})

# import pandas as pd
# import matplotlib.pyplot as plt
# df = pd.DataFrame(snapshots)
# df.hist()
# plt.show()


print('num snapshots', len(snapshots))
if len(snapshots) > 0:
    for snap in snapshots: # snapshots should have length 16 * 400 i.e. 6400
        # snapshot_response[0][2] -> cycle, [7] -> accound_id, [8] -> delegate_id, [9] -> is_delegate boolean,
        # [10] -> balance, [11] -> delegated_amount, [15] -> address, [16] -> delegate_address,
        snshot = Snapshot(snap[2], snap[7], snap[8], snap[9], snap[10], snap[11], snap[15], snap[16])
        insert_snapshot(snshot)

conn.commit()
