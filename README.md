# Model evaluation framework

## General info

The repo for the tool that evaluates models and gives pretty report

Please, write your suggestions, bug reports in the "Issues" section.
Pull requests are welcome.

The framework can be run in Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NickVeld/model_evaluation/blob/for_virus_models/model_evaluation.ipynb)

https://colab.research.google.com/github/NickVeld/model_evaluation/blob/for_virus_models/model_evaluation.ipynb

## Prediction format

Region,Country,TrialDate(M/D/YYYY),Infected_1,Died_1,Recovered_1,...,,Infected_n,Died_n,Recovered_n

where n - maximum gap between the trial and the prediction using it in days.

Time data in TrialDate will be removed.