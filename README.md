# Portfolio Optimization with Regularized Mean-Variance Model

<img src="images/stock_market.png" width="1000" >

I implement the Mean-Variance Optimization (MVO) model presented by [H. Markowitz (1952)](https://www.jstor.org/stable/2975974), on the fundamental concept that the optimal portfolio selection strategy is be an optimal trade-off between the return and risk. 

**Decision variables:**

$w = \\{w_1, w_2, ... , w_n \\}$: weight vector for all stocks. 

## The Model

**Parameters:**

$\lambda$: risk aversion. 

$\gamma$: regularization parameter for the L2 norm. 

$\mu$: expected return in percentage of stocks.

$\Sigma$: covariance matrix of all stocks.

**The MVO model:**

$$
\begin{align}
	\text{max}	& (1 - \lambda) \mu - \lambda (w^T \Sigma w) - \gamma \lVert w \rVert ^2 &\\    
	\text{s.t.} 	& \sum w = 1\\
    				& 0 \leq w \leq 1\\
\end{align}
$$

## The Code

The above mathematical model is encoded in Python Jupyter notebook with [CVXPY](https://www.cvxpy.org/) as the solver. Adding the following routine is necessary. 

Import libraries. 

```javascript
import cvxpy as cp
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
```
Import data. The Excel file "targetPrices.xlsx" should contain at least two columns, 'TICKER' and 'TARGET'. It should be noted that in this project, the target price of a stock is given as an input. Alternatively, the target price can also be calculated at a static fashion using standard models such as [Capital Asset Pricing Model (CAPM)](https://en.wikipedia.org/wiki/Capital_asset_pricing_model). 

```javascript
# The input is stock picks and their target prices from analysts
file_path = 'input_portf/targetPrices.xlsx'
myStocks = pd.read_excel(file_path, sheet_name='Sheet1', index_col='TICKER')[['TARGET']]
myStocks.sort_index(inplace=True)
```
Data-preprocessing. 

```javascript
# Get the stock tickers and target prices
tickers_list = myStocks.index.tolist()
targets = myStocks['TARGET'].to_numpy()

# Get stock prices from Yahoo Finance
data = yf.download(tickers_list, start = '2019-10-1', end = '2024-11-10')['Adj Close'].dropna(how="all")

prices = data.sort_index(axis=1)
print(prices)
```

```javascript
```

```javascript
```

```javascript
```

```javascript
```

```javascript
```

```javascript
```

```javascript
```

```javascript
```

```javascript
```

