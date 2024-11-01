# Portfolio Optimization with Regularized Mean-Variance Model


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
	\text{minimize:}	& \text{maximize} \\{0,\ u_{ik} - U + W - D_i \\} 	&\\    
	\text{subject to:} 	& \sum_{j \in s} x_{0jk} = 1, & \forall k \in \Theta 	\\
    				& \sum_{i \in s} x_{i0k} = 1,  &\forall k \in \Theta 	\\
   				& \sum_{j \in s} x_{ijk} = y_{ik}, & \forall i \in s, k \in \Theta	\\
\end{align}
$$

## The Code
