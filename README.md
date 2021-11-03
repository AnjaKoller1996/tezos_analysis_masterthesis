# Tezos Proof of Stake Fairness Analysis
This project contains a data analysis of the tezos blockchain rewards and rolls distribution among the bakers per cycle and blocks. This is an analysis regarding the fairness of the proof of stake mechanism of tezos. The code for the analysis implementation can be found under tezos_analysis.py and the database can be found in tezos_dataextraction_merged_alltables.db and under images one can find some plots.

# Questions
The following questions are addressed in this project: 
	1) Is the consensus protocol used by Tezos fair in terms of reward distribution? Does it lead to mechanisms 		   like rich-gets-richer?
	2) What is the effect on governance of the consensus algorithm?
	3) Are there statistical anomalies in the sequence of blocks created that may show adversarial attacks such 
	   such as selfish mining?

# API
The api used is Tezos tzstats api https://tzstats.com/docs/api

# Programming language
The programming language used is Python and to create and connect the SQL database sqlite3 is used.

# Measurements used
As an inequality measurement the Gini index is used.

# Method
The project contains the code to extract the data from the API and saving it in a DB and the code and plots of the analysis of the gini index computation or reward and rolls. TODO: add other stuff

# Analysis Results
TODO

# Challenges
A major challenge was that not all data could be gathered via the tzstats API. For a the distribution of rolls per baker we need a tezos node archive data which we collected using a tezos node as described in https://tezos.gitlab.io/introduction/howtoget.html#using-docker-images and accessed the data with https://tezos.gitlab.io/user/history_modes.html
