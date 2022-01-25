# Tezos Proof of Stake Fairness Analysis
This project contains a data analysis of the tezos blockchain rewards and rolls distribution among the bakers per cycle and blocks. This is an analysis regarding the fairness of the proof of stake mechanism of tezos. The code for the analysis implementation can be found under tezos_analysis.py and the database can be found in tezos_dataextraction_merged_alltables.db and under images one can find some plots.

# Questions
The following questions are addressed in this project: <br />
	1) Is the consensus protocol used by Tezos fair in terms of reward distribution? Does it lead to mechanisms like rich-gets-richer? <br />
	2) What is the effect on governance of the consensus algorithm?<br />
	3) Are there statistical anomalies in the sequence of blocks created that may show adversarial attacks such 
	   such as selfish mining?<br />

# API
The api used is Tezos tzstats api https://tzstats.com/docs/api

# Programming language
The programming language used is Python and to create and connect the SQL database sqlite3 is used.

# Measurements used
As an inequality measurement the Gini index, Expectational Fairness, Robust Fairness and Nakamoto index are used. 

# Method
The project contains the code to extract the data from the API and saving it in a DB and the code and plots for the different fairness measures. We look at the average, percentiles and mean values of these measurements and compare them along cycles and with values data from other blockchains. Further we compare the mean and average of rewards and take into account the major protocol upgrades of Tezos. 

# Analysis Results
Blockchains based on the Proof of Stake consensus protocol have gained increasing attention and various areas of applications in recent years. Previous research focuses on matters of energy consumption, decentralization and security of such blockchains. However, there is a lack of thorough studies concerning their fairness. Our contribution is to provide a careful investigation of fairness of the Tezos blockchain.
To this end, we look at the reward distribution, the number of participants and the stakes in the Tezos blockchain. Our analysis indicates that certain participants of the Tezos blockchain consensus mechanism receive disproportionately more rewards than others. Further, we find that the various Tezos protocol upgrades affect the reward distribution, meaning that the fairness of Tezos is impacted by its governance. 
To investigate the fairness in more detail, we apply various fairness measures such as the Gini index, Nakamoto index, expectational and robust fairness to the reward and stake distributions. These fairness measures confirm the previous observation that few participants in the Tezos blockchain receive a disproportionately large reward in comparison to their investments. Looking at these outliers, we are not able to attribute any particular strategies to their mining behavior. Nevertheless, they share the feature of having entered the system at a very early stage.
In terms of the aforementioned fairness measures we therefore consider the Tezos blockchain system unfair. In other words, we find that Tezos exhibits a rich-get-richer mechanism.

# Challenges and Limitations
A major challenge was that not all data could be gathered via the tzstats API. For a the distribution of rolls per baker we need a tezos node archive data which we collected using a tezos node as described in https://tezos.gitlab.io/introduction/howtoget.html#using-docker-images and accessed the data with https://tezos.gitlab.io/user/history_modes.html
Note, that we only the Tezos blockchain data from the start (cycle 0) to cycle 398 (August 2021) are considered for this analysis. Further, we do not investigate selfish mining detection. 

