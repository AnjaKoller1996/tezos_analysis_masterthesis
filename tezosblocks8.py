# -*- coding: utf-8 -*-
"""TezosBlocks8.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1b93uUv09QsPh_UwqQ8DocgTZmLkAzAx1
"""

# import requests
import json
import requests
from pprint import pprint
from datetime import datetime

# # create a DB with sqlite3
import sqlite3
conn = sqlite3.connect('tezos_blocks.db')
conn.isolation_level = None
c = conn.cursor() # db saved in location where .py file is save

class Block: 

  def __init__(self, cycle, baker, fee, reward, deposit):
    self.cycle = cycle
    self.baker = baker
    self.fee = fee
    self.reward = reward
    self.deposit = deposit

# insert blocks

def insert_block(block):
  with conn: 
    c.execute('INSERT INTO blocks VALUES(:cycle, :baker, :fee, :reward, :deposit)', {'cycle': block.cycle, 'baker': block.baker, 'fee': block.fee, 'reward': block.reward, 'deposit': block.deposit})

def insert_blocks(blocks):
  if len(blocks) > 0:
    for bl in blocks:
      block = Block(bl['cycle'], bl['baker'], bl['fee'], bl['reward'], bl['deposit'])
      insert_block(block)

conn.commit()

max_height = 1668702
url_template = 'https://api.tzstats.com/explorer/block/'

blocks = []
height = 1400000
# blocks with height 400000 to 699999
while (height < 1668702):
  block_to_insert = requests.get(url_template + str(height)).json()
  blocks.append(block_to_insert)
  height +=1

  # insert to db every 100 entries
  if height % 100 == 0:
    print('height', height)
    insert_blocks(blocks)
    blocks = []