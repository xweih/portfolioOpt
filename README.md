# Portfolio Optimization with Regularized Mean-Variance Model

<img src="images/stock_market.png" width="1000" >

I implement the Mean-Variance Optimization model presented by [H. Markowitz (1952)](https://www.jstor.org/stable/2975974), on the fundamental concept that the optimal portfolio selection strategy is be an optimal trade-off between the return and risk. 

## The Model

**Indicies and Sets:**

Locations: $S=\\{0,..., N \\}$, $s = \\{1,..., N \\}$

Trips: $\Theta =\\{1,..., K \\}$, $\theta =\\{2,..., K \\}$

Satellite: $i, j \in S$

Trip: $k \in \Theta $ 

**Decision variables:**

$x_{ijk} \in$ {0,1}: 1, if a walk from satellite i to j occurs in trip k, and 0, if not.  

**The MILP model:**

$$
\begin{align}
	\text{maximize:}	& (1-\lambda) \mu - \lambda w \Sigma w - \gamma \lVert x \rVert ^2 &\\    
	\text{subject to:} 	& \sum_{j \in s} x_{0jk} = 1, & \forall k \in \Theta 	\\
    				& \sum_{i \in s} x_{i0k} = 1,  &\forall k \in \Theta 	\\
   				& \sum_{j \in s} x_{ijk} = y_{ik}, & \forall i \in s, k \in \Theta	\\
\end{align}
$$

## The Code

The above mathematical model is encoded in Python Jupyter notebook with [CVXPY](https://www.cvxpy.org/) as the solver. Adding the following routine is necessary. 

```javascript

```
 
First, I preprocess the data, i.e., the satellites's locational information from a csv file. 
