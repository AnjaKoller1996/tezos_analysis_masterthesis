import sqlite3
import requests

conn = sqlite3.connect('income_data.db', timeout=10)
c = conn.cursor()

income_table_api_url = 'https://api.tzstats.com/tables/income'
incomes = []


def get_baker_addresses(offset):
    api_url_bakers = 'https://api.tzstats.com/explorer/bakers'
    params = {'limit': 1000, 'offset': offset}
    r = requests.get(api_url_bakers, params=params)
    baker_response = r.json()
    addresses = [r['address'] for r in baker_response]
    return addresses


def get_baker_address_list():
    """Needed as we only get portions of 100 bakers per request"""
    addresses_list = get_baker_addresses(0)
    for addr in get_baker_addresses(100):
        addresses_list.append(addr)
    for addr in get_baker_addresses(200):
        addresses_list.append(addr)
    for addr in get_baker_addresses(300):
        addresses_list.append(addr)
    print(' length address list', len(addresses_list))
    # remove duplicates
    addresses_list = list(dict.fromkeys(addresses_list))
    print(' length address list no duplicates', len(addresses_list))
    return addresses_list


baker_addresses = get_baker_address_list()
print('number of addresses', len(baker_addresses))

for addr in baker_addresses:
    params = {'limit': 10000}
    r = requests.get(income_table_api_url + '?address=' + addr,
                     params=params)
    snapshot_response = r.json()
    incomes.extend(snapshot_response)

""" Arguments we want:
    row_id: unique row identifier x[0]
    cycle: cycle this income relates to x[1]
    rolls: number of rolls at snapshot block x[3]
    balance: a bakers own balance (spendable balance and frozen deposits and frozen fees), at snapshot block x[4]
    n_baking_rights: count of baking rights in this cycle x[7]
    n_blocks_baked: number of blocks baked in this cycle x[13]
    expected_income: total income based on endorsing and priority 0 baking rights x[19]
    total_income: total sum of all income x[21]
    baking_income: total income from baking blocks x[23]
    fees_income: total income from fees x[28]
    address: Account address --> query by baker addresses x[38]
"""


class Income:
    def __init__(self, row_id, cycle, rolls, balance, n_baking_rights, n_blocks_baked, expected_income, total_income,
                 baking_income,
                 fees_income, address):
        self.row_id = row_id
        self.cycle = cycle
        self.rolls = rolls
        self.balance = balance
        self.n_baking_rights = n_baking_rights
        self.n_blocks_baked = n_blocks_baked
        self.expected_income = expected_income
        self.total_income = total_income
        self.baking_income = baking_income
        self.fees_income = fees_income
        self.address = address


# create an income table
c.execute(''' SELECT count(name) from sqlite_master where type='table' AND name='income_table' ''')

if c.fetchone()[0] == 1:
    c.execute('DROP TABLE income_table')
    c.execute(
        '''CREATE TABLE income_table (row_id integer, cycle integer, rolls integer, balance real, n_baking_rights 
        integer, n_blocks_baked integer, expected_income real, total_income real, baking_income real, fees_income 
        real, address text)''')
else:
    c.execute(
        '''CREATE TABLE income_table (row_id integer, cycle integer, rolls integer, balance real, n_baking_rights 
        integer, n_blocks_baked integer, expected_income real, total_income real, baking_income real, fees_income 
        real, address text)''')


def insert_income(inc):
    with conn:
        c.execute(
            'INSERT INTO income_table VALUES(:row_id, :cycle, :rolls, :balance, :n_blocks_baked, :expected_income, '
            ':n_baking_rights, :total_income, :baking_income, :fees_income, :address)',
            {'row_id': inc.row_id, 'cycle': inc.cycle, 'rolls': inc.rolls, 'balance': inc.balance,
             'n_baking_rights': inc.n_baking_rights, 'n_blocks_baked': inc.n_blocks_baked,
             'expected_income': inc.expected_income, 'total_income': inc.total_income,
             'baking_income': inc.baking_income, 'fees_income': inc.fees_income, 'address': inc.address})


if len(incomes) > 0:
    for income in incomes:
        inc = Income(income[0], income[1], income[3], income[4], income[7], income[13], income[19], income[21],
                     income[23], income[28], income[38])
        insert_income(inc)

print('Done')

conn.commit()
