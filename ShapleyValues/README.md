# The Shapley Value
- comes from game theory to answer the question: What is the *fair* way to asses the contributions of some coalition of agents $S \subseteq {1,... N}$ to some value $v$? In other words, what is the best way to divide the credit/payout for a group project? 
- **Approach**: members contribute value according to their *marginal contribution*
- To identify what we mean by *fair*, we take an axiomatic approach 
## Game theory Axioms for Shapley Values
- **Definition** two agents $i$ and $j$ are called *interchangeable* if for every $S$ containing neither $i$ or $j$, we have 
\begin{equation}
    v\left( S \cup \{i\} \right) = v\left( S \cup \{j\} \right)
\end{equation}

1. **(Symmetry)** For any $v$, if $i$ and $j$ are interchangeable, then 
\begin{equation}
    \psi_i(N, v) = \psi_j(N, v)
\end{equation}
where $\psi_i$ and $\psi_j$ are the payout agents $i$ and $j$ receive. 
2. **(Dummy players)** If the amount that $i$ contributes to any coalition is $0$, $i$ is called a dummy player. In other words $\forall S$,
\begin{equation}
    v\left( S \cup \{ i\} \right) = v\left(S\right)
\end{equation}
For any such *dummy player*, the payout is *nothing*, i.e. $\psi_i(N, v) = 0$
3. **(Additivity)** If we can split a game into two parts $v = v_1 + v_2$, then we can decompose the payouts, i.e. if $(v_1+v_2)(S) = v_1(S) + v_2(S)$ for every $S$, then for each $i$, 
\begin{equation}
    \psi_i(N, v_1+v_2) = \psi_i(N, v_1) + \psi_i(N, v_2)
\end{equation}

## The Shapley Value
Given a coalitional game $(N, v)$, the *Shapley Value* divides the payoffs among players according to: 
\begin{equation}
    \phi_i(N, v) = \frac{1}{N!} \sum_{S\subseteq N\setminus \{i\}} |S|!\Big(|N| - |S| - 1 \Big)\Big[ v\left( S \cup \{i\} \right) - v(S) \Big]
\end{equation}
where 
- $|S|!$ is the number of ways the set $S$ could have been formed prior to $i$'s addition
- $(|N|-|S|-1)!$ is the number of ways the remaining player could be added. 
- $|N|!$ is the total number of different orderings we could have for $N$

**Theorem**: Given a coalitional game $(N, v)$, there is a unique payoff division that satisfies the axioms: the Shapely value.

## Examples
### 1. Two Partners Sharing Profits
Suppose we have $v(\{1\} = 1$, $v(\{2\}) = 2$, but $v(\{1,2\}) = 4$. In other words, $1$ and $2$ can achieve more value together but do not have the same value individually. We could have built the coalition two ways: 
- $\{1\} \rightarrow \{1, 2\} $ so that $v(1) = 1$
- $\{2\} \rightarrow \{1, 2\} $ so that $v(1) = v(12) - v(2) = 2$
There are two options, so each gets a weight of $1/2$ so that 
\begin{align}
    \phi_1 &= 1.5 \\ 
    \phi_2 &= 2.5
\end{align}

## Estimating the Shapely Value for real models
[link](https://christophm.github.io/interpretable-ml-book/shapley.html)

As we add more features (i.e. players in the game scheme) to our model, the number of coalitions quickly exceeds numerical convenience. Therefore, we use a Monte-Carlo sampling scheme to approximate the SHAP values
\begin{equation}
    \hat{\phi}_j = \frac{1}{M}\sum_{m=1}^M \left( \hat{f}(x^m_{+j})- \hat{f}(x^m_{-j}) \right)
\end{equation}
In pseudo-code this is: 
For $m\in 1, ... , M$
1. Draw a random instance $z$ from the data matrix $X$ 
2. choose a random permutation $\pi$ of the feature columns. We now have two instances $x$, and $z$ with 
   - $x_\pi = (x_{\pi_1}, ..., x_{\pi_j}, ..., x_{\pi_p})$
   - $z_\pi = (z_{\pi_1}, ..., z_{\pi_j}, ..., z_{\pi_p})$
3. Construct two new instances: 
   - With $j$: $x_{+j} = (x_{\pi_1}, ..., x_{\pi_{j-1}}, x_{\pi_j}, z_{\pi_{j+1}}, ..., z_{\pi_p})$
   - Without $j$: $x_{-j} = (x_{\pi_1}, ..., x_{\pi_{j-1}}, z_{\pi_j}, z_{\pi_{j+1}}, ..., z_{\pi_p})$
4. Compute the marginal contribution: 
\begin{equation}
    \phi_j^m = \hat{f}(x_{+j}) - \hat{f}(x_{-j})
\end{equation}
5. Compute average marginal contribution across all $M$ iterations
\begin{equation}
    \phi_j(x) = \frac{1}{M} \sum_{m=1}^M \phi_j^m
\end{equation}
6. Repeat the procedure for each of the $p$ features.
