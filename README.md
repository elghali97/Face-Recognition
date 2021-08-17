# Deep learning for large picture database indexing

## Dataset

Complete dataset can be found here : https://fex.insa-lyon.fr/get?k=DMBBtY0Z90zvqMtRqOZ (expired the 12/12/2019)

This dataset countains both training and test dataset for a total of 163 100 pictures (78 800 face and 84 300 non-face) in 36x36 format.

In order to work correctly, it needs to be extract at the root of the project.

## Exploration

`Notebook.ipnyb` is the notebook in which we have made our experiments to build our model. All the process is detailled in the notebook.

## Processing

`Training.py` and `Visualizator.py` are the two main scripts of this project.

`Training.py` is intended to train networks and save the best one in the form of a k-fold cross validation. You can train multiple network structure by adding your own network in the `Net` module and adding your network class in the `net_type` array. You can set various parameters like number of epoch, validation split etc... For complete the list of options, please refer to the `-h` option.

`Visualizator.py` is intended to see and store results of face recognition on a picture. Options such as threshold, number of minimal votes can be set (Complete list can be found with `-h` option).

## Results
Couple of results can be seen in the `results` folder. By default, result after visualization are stored in this folder.

