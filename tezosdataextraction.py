import requests
from pprint import pprint

"""**TODO**: create the following table only using tzstats
--> **Account, Rewards, Bakers, Blocks, Contracts, Cycles**

**SQLite3 Database**(see: https://datatofish.com/)

1.   Connect your database name
2.   Create DB and Tables
3.   Import data using pandas


create-database-python-using-sqlite3/)
"""

# create a DB with sqlite3
import sqlite3

conn = sqlite3.connect('tezosdataextraction.db', timeout=10)
conn.isolation_level = None
c = conn.cursor()  # db saved in location where .py file is save

"""# 1) Bakers"""
# Bakers
api_url_bakers = 'https://api.tzstats.com/explorer/bakers'
params = {'limit': 10000}
r = requests.get(api_url_bakers, params=params)
baker_response = r.json()


class Baker:

    def __init__(self, address, id, total_balance, staking_balance, staking_capacity):
        self.address = address
        self.id = id
        self.total_balance = total_balance
        self.staking_balance = staking_balance
        self.staking_capacity = staking_capacity


# Create baker table
c.execute(''' SELECT count(name) FROM sqlite_master where type='table' AND name= 'bakers' ''')

if c.fetchone()[0] == 1:
    c.execute('DROP TABLE bakers')
    c.execute(
        '''CREATE TABLE bakers (address text, id integer, total_balance integer, staking_balance integer, 
        staking_capacity integer)''')
else:
    c.execute(
        '''CREATE TABLE bakers (address text, id integer, total_balance integer, staking_balance integer, 
        staking_capacity integer)''')


def insert_baker(baker):
    with conn:
        c.execute('INSERT INTO bakers VALUES(:address, :id, :total_balance, :staking_balance, :staking_capacity)',
                  {'address': baker.address, 'id': baker.id, 'total_balance': baker.total_balance,
                   'staking_balance': baker.staking_balance, 'staking_capacity': baker.staking_capacity})


# add all bakers to db
for baker in baker_response:
    baker = Baker(baker['address'], baker['id'], baker['total_balance'], baker['staking_balance'],
                  baker['staking_capacity'])
    insert_baker(baker)

conn.commit()

"""# 2) Accounts"""

# Accounts (get accounts by baker addresses)
api_url_accounts = 'https://api.tzstats.com/explorer/account/'

# Get all account bakers (rewards --> total_rewards_earned)
addresses = [r['address'] for r in baker_response]
account_response = []

for address in addresses:
    r = requests.get(api_url_accounts + address)
    account_response.append(r.json())

num_account_baker_responses = len(account_response)
account_baker_rewards = [r['total_rewards_earned'] for r in account_response]


class Account:

    def __init__(self, address, delegate, total_rewards_earned, total_fees_earned, total_lost, total_balance,
                 staking_balance):
        self.address = address
        self.delegate = delegate
        self.total_rewards_earned = total_rewards_earned
        self.total_fees_earned = total_fees_earned  # this here we are really interested in
        self.total_lost = total_lost
        self.total_balance = total_balance
        self.staking_balance = staking_balance


# Create accounts
c.execute(''' SELECT count(name) from sqlite_master where type='table' AND name='accounts' ''')

if c.fetchone()[0] == 1:
    c.execute('DROP TABLE accounts')
    c.execute(
        '''CREATE TABLE accounts (address text, delegate text, total_rewards_earned integer, total_fees_earned 
        integer, total_lost integer, total_balance integer, staking_balance integer)''')
else:
    c.execute(
        '''CREATE TABLE accounts (address text, delegate text, total_rewards_earned integer, total_fees_earned 
        integer, total_lost integer, total_balance integer, staking_balance integer)''')


def insert_acc(accbaker):
    with conn:
        c.execute(
            'INSERT INTO accounts VALUES(:address, :delegate, :total_rewards_earned, :total_fees_earned, :total_lost, '
            ':total_balance, :staking_balance)',
            {'address': accbaker.address, 'delegate': accbaker.delegate,
             'total_rewards_earned': accbaker.total_rewards_earned, 'total_fees_earned': accbaker.total_fees_earned,
             'total_lost': accbaker.total_lost, 'total_balance': accbaker.total_balance,
             'staking_balance': accbaker.staking_balance})


# add all bakers to db
for accbak in account_response:
    accbaker = Account(accbak['address'], accbak['delegate'], accbak['total_rewards_earned'],
                       accbak['total_fees_earned'], accbak['total_lost'], accbak['total_balance'],
                       accbak['staking_balance'])
    insert_acc(accbaker)

conn.commit()

"""# 3) Cycles"""

# Cycles
api_url_cycles = 'https://api.tzstats.com/explorer/cycle/head'
r = requests.get(api_url_cycles)
cycles_response = r.json()


class Cycle:

    def __init__(self, cycle, snapshot_index, active_bakers, active_delegators):
        self.cycle = cycle
        self.snapshot_index = snapshot_index
        self.active_bakers = active_bakers
        self.active_delegators = active_delegators


# Create cycles table: check if cycles table already exist
c.execute(''' SELECT count(name) FROM sqlite_master where type='table' AND name= 'cycles' ''')

# if count 1 then table already exist
if c.fetchone()[0] == 1:
    c.execute('DROP TABLE cycles')
    c.execute(
        '''CREATE TABLE cycles (cycle integer, snapshot_index integer, active_bakers integer, active_delegators 
        integer)''')
else:
    c.execute(
        '''CREATE TABLE cycles (cycle integer, snapshot_index integer, active_bakers integer, active_delegators 
        integer)''')


def insert_cycle(cycle):
    with conn:
        c.execute('INSERT INTO cycles VALUES(:cycle, :snapshot_index, :active_bakers, :active_delegators)',
                  {'cycle': cycle['cycle'], 'snapshot_index': cycle['snapshot_index'],
                   'active_bakers': cycle['active_bakers'], 'active_delegators': cycle['active_delegators']})


# add cycles to db
cycle = Cycle(cycles_response['cycle'], cycles_response['snapshot_index'], cycles_response['active_bakers'],
              cycles_response['active_delegators'])
insert_cycle(cycles_response)

conn.commit()

"""# 4) Contracts"""

# Contracts
api_url_contracts = 'https://api.tzstats.com/explorer/contracts/'
# get contract addresses from accounts where is_contract=True
contract_accounts = []
print(account_response[0]['is_contract'] == True)

for acc in account_response:
    if acc['is_contract']:
        contract_accounts.append(acc)

# contract_accounts = account_response['is_contract'] == True
print(contract_accounts)
if len(contract_accounts) > 0:
    contract_addresses = [r['address'] for r in contract_accounts]
    contracts_response = []
    for address in contract_addresses:
        r = requests.get(api_url_contracts + address)
        contracts_response.append(r.json())


class Contract:

    def __init__(self, address, creator, delegate):
        self.address = address
        self.creator = creator
        self.delegate = delegate


# Create contracts
c.execute(''' SELECT count(name) from sqlite_master where type='table' AND name='contracts' ''')

if c.fetchone()[0] == 1:
    c.execute('DROP TABLE contracts')
    c.execute('''CREATE TABLE contracts (address text, creator text, delegate text)''')
else:
    c.execute('''CREATE TABLE contracts (address text, creator text, delegate text)''')


def insert_contract(contract):
    with conn:
        c.execute('INSERT INTO contracts VALUES(:address, :creator, :delegate)',
                  {'address': contract.address, 'creator': contract.creator, 'delegate': contract.delegate})


# add contract to db
if len(contract_accounts) > 0:
    contract = Contract(contracts_response['address'], contracts_response['creator'], contracts_response['delegate'])
    insert_contract(contract)

conn.commit()

"""# 5) Delegator"""

# Delegates & their rewards and fees
delegators = []
for acc in account_response:
    if acc['is_delegate']:
        delegators.append(acc)

num_delegators = len(delegators)
print('Number of delegators', num_delegators)


class Delegator:

    def __init__(self, address, total_rewards_earned, total_fees_earned, total_fees_paid):
        self.address = address
        self.total_rewards_earned = total_rewards_earned
        self.total_fees_earned = total_fees_earned
        self.total_fees_paid = total_fees_paid


# create a delegator table
c.execute(''' SELECT count(name) from sqlite_master where type='table' AND name='delegators' ''')

if c.fetchone()[0] == 1:
    c.execute('DROP TABLE delegators')
    c.execute(
        '''CREATE TABLE delegators (address text, total_rewards_earned text, total_fees_earned text, total_fees_paid 
        text)''')
else:
    c.execute(
        '''CREATE TABLE delegators (address text, total_rewards_earned text, total_fees_earend text, total_fees_paid 
        text)''')


def insert_delegator(delegator):
    with conn:
        c.execute(
            'INSERT INTO delegators VALUES(:address, :total_rewards_earned, :total_fees_earned, :total_fees_paid)',
            {'address': delegator.address, 'total_rewards_earned': delegator.total_rewards_earned,
             'total_fees_earned': delegator.total_fees_earned, 'total_fees_paid': delegator.total_fees_paid})


# add contract to db
if len(delegators) > 0:
    for deleg in delegators:
        delegator = Delegator(deleg['total_rewards_earned'], deleg['total_fees_earned'], deleg['delegate'],
                              deleg['total_fees_paid'])
        insert_delegator(delegator)

conn.commit()


class Block:

    def __init__(self, cycle, baker, fee, reward, deposit):
        self.cycle = cycle
        self.baker = baker
        self.fee = fee
        self.reward = reward
        self.deposit = deposit


url_template = 'https://api.tzstats.com/explorer/block/'
block_api_url = 'https://api.tzstats.com/explorer/block/head'
all_blocks = []
block = requests.get(block_api_url).json()
all_blocks.append(block)


while block.get('predecessor', None) is not None:
    predecessor_hash = block['predecessor']
    predecessor_block = requests.get(url_template + predecessor_hash).json()
    all_blocks.append(predecessor_block)
    block = predecessor_block

# create blocks table
c.execute(''' SELECT count(name) from sqlite_master where type='table' AND name='blocks' ''')

if c.fetchone()[0] == 1:
    c.execute('DROP TABLE blocks')
    c.execute('''CREATE TABLE blocks (cycle integer, baker text, fee real, reward real, deposit integer)''')
else:
    c.execute('''CREATE TABLE blocks (cycle integer, baker text, fee real, reward real, deposit integer)''')


def insert_block(block):
    with conn:
        c.execute('INSERT INTO blocks VALUES(:cycle, :baker, :fee, :reward, :deposit)',
                  {'cycle': block.cycle, 'baker': block.baker, 'fee': block.fee, 'reward': block.reward,
                   'deposit': block.deposit})


if len(all_blocks) > 0:
    for bl in all_blocks:
        block = Block(bl['cycle'], bl['baker'], bl['fee'], bl['reward'], bl['deposit'])
        insert_block(block)

conn.commit()
