# Maximize Sharpe Ratio Strategy

## Repository organization

- [README.md](README.md): Introduces the repository
- [strategy](strategy/): strategy module
- [main.ipynb](main.ipynb): Jupyter notebook with documentation of Ichimoku Cloud strategy, execution of the strategy and report generation
- [data_market](data_market/): data provider module
- [datasets](datasets/): subdirectory with the data made available for the strategy
- [libs](libs/): additional external libraries
- [simulator](simulator/): support code for simulating the strategy
- [requirements.txt](requirements.txt): dependent libraries

## How to run

1. Use virtual environment (more help: [The Hitchhiker's Guide to Python](https://docs.python-guide.org/dev/virtualenvs/))
```bash
pipenv install -r requirements.txt
pipenv shell
```
- Important because the QuantStat library is a modified version
2. Open `main.ipynb` in Jupyter Notebook
The `main.ipynb` notebook contains:

- a brief explanation of the strategy;
- the execution of the strategy;
- a report to compare the strategy with a benchmark.

Note that the strategy is not implemented in the notebook. We only invoke it in the notebook to keep the code organized.

The strategy is contained in the `strategy` folder, as well as all the auxiliary functions for its execution.

Before running any strategy, we need to load the data. It is preferable to concentrate the data processing separately from the strategy. Here we store the data and functions in the `data_market` folder.

Finally, the notebook will open a window asking where to save the report. Save it in a folder and open it in a browser. You can compare your strategy with any ticker compatible with Yahoo Finance.