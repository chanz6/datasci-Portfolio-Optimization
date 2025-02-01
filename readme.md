# Portfolio Optimization and Risk Analysis

## Introduction
In this project, I will analyze my stock portfolio and apply gradient ascent to optimize it. All stock data will be sourced from Yahoo Finance, with the analysis code stored in `analysis.ipynb` and the gradient ascent algorithm implemented in `main.ipynb`.

I will utilize various risk analytics techniques in Python, leveraging libraries such as Pandas, NumPy, Matplotlib, and Seaborn. My approach involves analyzing my current portfolio, optimizing it using gradient ascent, and evaluating the optimized portfolio to assess how the algorithm impacts key performance metrics.

Here is my current portfolio, which we will optimize by adjusting the weight distribution:

_Weight represents the percentage of my TFSA account allocated to each stock._

| Stock | Weight |
|-|-|
| AMZN | 19.83% |
| GOOGL | 28.45% |
| MSFT | 14.47% |
| NVDA | 8.37% |
| VOO | 28.88% |

---

## Shortcuts

**1. [Calculating the Sharpe Ratio](https://github.com/chanz6/Portfolio-Optimization-Risk-Analysis?tab=readme-ov-file#the-sharpe-ratio-and-its-components)**
- 1.1 [Definition of the Sharpe Ratio](https://github.com/chanz6/Portfolio-Optimization-Risk-Analysis?tab=readme-ov-file#1-definition-of-the-sharpe-ratio)
- 1.2 [Expected Return of an Asset](https://github.com/chanz6/Portfolio-Optimization-Risk-Analysis?tab=readme-ov-file#2-expected-return-of-an-asset)
- 1.3 [Expected Portfolio Return](https://github.com/chanz6/Portfolio-Optimization-Risk-Analysis?tab=readme-ov-file#3-expected-portfolio-return)
- 1.4 [Volatility of an Asset](https://github.com/chanz6/Portfolio-Optimization-Risk-Analysis?tab=readme-ov-file#4-volatility)
- 1.5 [Portfolio Volatility](https://github.com/chanz6/Portfolio-Optimization-Risk-Analysis?tab=readme-ov-file#5-portfolio-volatility)
- 1.6 [Computing the Sharpe Ratio](https://github.com/chanz6/Portfolio-Optimization-Risk-Analysis?tab=readme-ov-file#6-computing-the-sharpe-ratio)

**2. [Sharpe Ratio Optimizer](https://github.com/chanz6/Portfolio-Optimization-Risk-Analysis?tab=readme-ov-file#optimizing-the-sharpe-ratio)**
- 2.1 [Gradient of the Sharpe Ratio](https://github.com/chanz6/Portfolio-Optimization-Risk-Analysis?tab=readme-ov-file#1-gradient-of-the-sharpe-ratio)
- 2.2 [Optimizing Using Gradient Ascent](https://github.com/chanz6/Portfolio-Optimization-Risk-Analysis?tab=readme-ov-file#2-optimization-using-gradient-ascent)

**3. [Analysis](https://github.com/chanz6/Portfolio-Optimization-Risk-Analysis?tab=readme-ov-file#portfolio-analysis)**
- 3.1 [Before Optimization](https://github.com/chanz6/Portfolio-Optimization-Risk-Analysis?tab=readme-ov-file#1-before-optimization)
- 3.2 [Exploring Portfolio Risk and Return](https://github.com/chanz6/Portfolio-Optimization-Risk-Analysis?tab=readme-ov-file#exploring-portfolio-risk-and-return)
- 3.3 [After Optimization](https://github.com/chanz6/Portfolio-Optimization-Risk-Analysis?tab=readme-ov-file#2-after-optimization)

**4. [Summary of Results / Conclusion](https://github.com/chanz6/Portfolio-Optimization-Risk-Analysis?tab=readme-ov-file#conclusion)**

---

## The Sharpe Ratio and its Components
There are many ways to measure how good a portfolio is, but I’ve chosen to optimize the **Sharpe Ratio**. The Sharpe Ratio helps compare different investments by showing how much return you’re getting for each unit of risk taken. A higher Sharpe Ratio means better risk-adjusted returns, meaning the portfolio is earning more relative to how much it fluctuates. 

I chose this metric because portfolio volatility is almost always inevitable, so it makes sense to leverage this and aim for the best possible balance between risk and return.

### 1. Definition of the Sharpe Ratio

The Sharpe Ratio is defined as follows:

$$\text{Sharpe Ratio} = \frac{\text{Portfolio Return - Risk Free Rate}}{\text{Portfolio Volatility}}$$

This formula tells us how much excess return the portfolio generates for each unit of risk taken.

### 2. Expected Return of an Asset:

The expected return of an asset measures the average return of an asset over time. It is calculated in two steps:

**Step 1: Daily Expected Return**:

$$E(R_{daily}) = \frac{\displaystyle\sum R_t}{n}$$

where:
- $R_t$ = Return on day t
- $n$ = Total number of days

**Step 2: Annual Expected Return** (assuming 252 trading days in a year):

$$E(R_{annual}) = (1 + E(R_{daily}))^{252} - 1$$

**Code Implementation** (`main.ipynb`):

```
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
```
```
# import data & add daily returns column
def import_data(stocks):
    stocks_data = {}
    for stock in stocks:
        stock_data = yf.download(stock, start='2010-01-01', end='2025-01-01')
        stock_data['Return'] = stock_data['Close'].pct_change()
        stocks_data[stock] = stock_data
    return stocks_data
```
```
# calculate annual expected return
def annual_expected_return(data):
    results_dict = {}
    # calculate expected daily return
    returns_data = pd.DataFrame({stock: data[stock]['Return'] for stock in data.keys()})
    e_daily_returns = np.sum(returns_data, axis=0) / len(returns_data)
    # calculate expected annual return
    e_annual_returns = (1 + e_daily_returns)**252 - 1
    # record results
    results_dict['Daily Returns'] = e_daily_returns
    results_dict['Annual Returns'] = e_annual_returns

    return results_dict
```

---

### 3. Expected Portfolio Return
Once we have the expected annual return of each stock, we can calculate the expected return of the entire portfolio as follows:

$$E(R_p) = \displaystyle\sum (\omega_i \times E(R_i))$$
where:
- $E(R_p)$ = Expected return of the portfolio (in 1 year)
- $\omega_i$ = Weight (proportion) of asset $i$ in the portfolio
- $E (R_i)$ = Expected return of asset $i$

**Code Implementation** (`main.ipynb`):

```
def portfolio_return(data, weights):
    # extract expected annual returns
    e_annual = annual_expected_return(data)
    e_annual_returns = np.array(e_annual['Annual Returns'].values)
    # compute expected portfolio return
    return sum(weights[i] * e_annual_returns[i] for i in range(len(weights)))
```

---

### 4. Volatility

Volatility measures how much an asset's price fluctuates over time. 

**Annualized Volatility of an Asset**:

The annualized volatility for an asset $i$ is calculated as follows _(assuming 252 trading days)_:

$$\sigma_{annual} = \sigma_{daily} \times \sqrt{252}$$

where daily volatility ($\sigma_{daily}$) is defined as:

$$\sigma_{daily} = \sqrt{\frac{\displaystyle\sum (R_t - \hat{R})^2}{n-1}}$$

where:
- $R_t$ = Daily return on day $t$
- $\hat{R}$ = Average daily return
- $n$ = Number of trading days

**Code Implementation** (`main.ipynb`):

```
# calculate annualized volatility
def annualized_volatility(data):
    results_dict = {}
    # compute average daily returns
    returns_data = pd.DataFrame({stock: data[stock]['Return'] for stock in data.keys()})
    # compute standard deviation of daily returns
    volatility_daily = returns_data.std()
    # compute annualized volatility
    volatility_annualized = volatility_daily * np.sqrt(252)
    # record results
    results_dict['Daily Volatility'] = volatility_daily
    results_dict['Anual Volatility'] = volatility_annualized

    return results_dict
```

---

### 5. Portfolio Volatility

Portfolio volatility ($\sigma_p$) is calculated as follows:

$$\sigma_p = \sqrt{\omega^T\gamma\omega}$$

where:
- $\omega$ = Portfolio weights (vector of size $n$)
- $\gamma$ = Annualized covariace matrix (constructed using annualized volatilities and correlations)
- $\omega^T$ = Transpose of weights vector

**Code Implementation** (`main.ipynb`):

```
# calculate portfolio volatility
def portfolio_volatility(data, weights):
    returns_data = pd.DataFrame({stock: data[stock]['Return'] for stock in data.keys()})
    returns_data.dropna(inplace=True)
    # compute covariance matrix
    daily_cov = np.cov(returns_data, rowvar=False)
    annualized_cov = daily_cov * 252
    # convert weights & volatilities to arrays
    w = np.array(weights)
    wT = np.transpose(w)
    # compute portfolio volatility
    return np.sqrt(np.dot(wT, np.dot(annualized_cov, w)))
```

---

### 6. Computing the Sharpe Ratio

Now that we've computed both expected portfolio return and portfolio volatility, we can finally compute the Sharpe Ratio:

$$\text{Sharpe Ratio} = \frac{E(R_p) - R_f}{\sigma_p}$$

where:
- $E(R_p)$ = Expected portfolio return
- $R_f$ = Risk-free rate (assumed 4% based on U.S. Treasury Yield)
- $\sigma_p$ = Portfolio Volatility

**Code Implementation** (`main.ipynb`):

```
def sharpe_ratio(data, weights):
    results_dict = {}
    p_return = portfolio_return(data, weights)
    rfr = 0.04 # 4% (U.S. Treasury Yield)
    p_volatility = portfolio_volatility(data, weights)
    # compute sharpe ratio
    return (p_return - rfr) / p_volatility
```

---

## Optimizing the Sharpe Ratio

### 1. Gradient of the Sharpe Ratio

We wish to find the gradient of the sharpe ratio formula.

**Sharpe Ratio Formula**:

$$\text{Sharpe Ratio} = \frac{E(R_p) - R_f}{\sigma_p}$$

where:
- $E(R_p) = w^T\mu$ (Expected Portfolio Return)
- $R_f =$ risk-free rate (assumed to be 0.04)
- $\sigma_p = \sqrt{w^T\gamma w}$ (Portfolio Volatility)
- $w =$ Vector of portfolio weights
- $\mu =$ Vector of expected returns
- $\gamma =$ Covariance matrix of stock returns

We begin by computing partial derivatives.

**Gradient of Portfolio Return $E(R_p)$**:

$$\frac{\partial E(R_p)}{\partial w} = \mu$$

**Gradient of Portfolio Volatility $\sigma_p$**:

$$\frac{\partial\sigma_p}{\partial w} = \frac{\gamma w}{\sigma_p}$$

Thus we have,

**Gradient of the Sharpe Ratio**:

$$\nabla SR = \frac{\frac{\partial E(R_p)}{\partial w}\cdot\sigma_p - E(R_p)\cdot\frac{\partial\sigma_p}{\partial w}}{\sigma_p^2} = \frac{\mu\sigma_p - E(R_p)\cdot\frac{\gamma w}{\sigma_p}}{\sigma_p^2}$$

**Code Implementation** (`main.ipynb`):

```
def sharpe_ratio_gradient(data, weights):
    p_return = portfolio_return(data, weights)
    p_volatility = portfolio_volatility(data, weights)
    # compute mu (vector of expected returns)
    mu = np.array(annual_expected_return(data)['Annual Returns'].values)
    # compute gamma (covariance matrix of stock returns)
    returns_data = pd.DataFrame({stock: data[stock]['Return'] for stock in data.keys()})
    returns_data.dropna(inplace=True)
    daily_cov = np.cov(returns_data, rowvar=False)
    gamma = daily_cov * 252
    # compute gradient
    dE = mu
    dsigma = np.dot(gamma, weights) / p_volatility
    grad_sharpe = (dE * p_volatility - p_return * dsigma) / (p_volatility)**2

    return grad_sharpe
```

---

### 2. Optimization Using Gradient Ascent

For portfolio optimization, we use a gradient ascent algorithm to maximize the Sharpe Ratio by iteratively adjusting asset allocations.

1. **Gradient Calculation**: The algorithm computes the gradient at the current `weights` position, indicating the direction of the steepest increase in Sharpe Ratio.
2. **Regularization**: To encourage portfolio diversification, an L2 regularization term is applied to the gradient. Without this, the algorithm may exploit the Sharpe Ratio formula by concentrating allocations in high-return assets while disregarding lower-risk investments.
3. **Weight Update**: The portfolio weights are adjusted in the direction of the gradient to improve the Sharpe Ratio.
4. **Projection**: The updated weights are projected onto the simplex to ensure all values remain **non-negative** and **sum to 1**, further promoting a well-diversified portfolio.
5. **Iterative Optimization**: Steps 1-4 are repeated for `num_iterations` until weights converge to an optimal Sharpe Ratio.

**Additional Features**:
- **Step Size Decay**: The learning rate (`alpha`) is reduced by a factor of `decay` after each iteration, improving convergence stability and preventing overshooting.
- **Early Stopping Condition**: If the Sharpe Ratio decreases for **5 consecutive iterations**, the algorithm terminates early to prevent unnecessary computations and potential degradation of results.

**Code Implementation** (`main.ipynb`):

```
# projection to simplex (ensure all weights are non-negative and sum to 1)
def project_to_simplex(weights, epsilon=1e-4, lambda_reg=0.01):
    # Ensure weights have a minimum value to prevent zero allocations
    weights = np.maximum(weights, epsilon)

    # Apply L2 regularization to prevent extreme allocations
    weights -= lambda_reg * weights

    # Sort weights in descending order for projection
    sorted_weights = np.sort(weights)[::-1]
    cumulative_sum = np.cumsum(sorted_weights) - 1  # Adjust for sum constraint

    # Find the largest rho index that satisfies the condition
    rho = np.where(sorted_weights - (cumulative_sum / (np.arange(len(weights)) + 1)) > 0)[0][-1]
    lambda_ = cumulative_sum[rho] / (rho + 1)

    # Project weights onto the simplex while ensuring a minimum threshold
    projected_weights = np.maximum(weights - lambda_, epsilon)

    # Normalize to sum to 1
    return projected_weights / np.sum(projected_weights)
```

```
# implement gradient ascent to maximize the Sharpe Ratio
def portfolio_optimization(data, weights, alpha=0.5, decay=0.99, num_iterations=200, lambda_reg=0.05, patience=7):
    recent_sharpe = -np.inf
    counter = 0
    results = {'Iteration': [0], 
               'Sharpe Ratio': [sharpe_ratio(data, weights)],
               'Weights': [weights]
    }
    for i in range(num_iterations):
        grad_sharpe = sharpe_ratio_gradient(data, weights)
        # apply regularization to avoid extreme weights
        grad_sharpe -= lambda_reg * weights
        # update gradient ascent
        weights += alpha * grad_sharpe
        # projection with bias to prevent zero weights
        weights = project_to_simplex(weights)
        # decay
        alpha *= decay
        # print results
        sharpe = sharpe_ratio(data, weights)
        # print(weights)
        # print(sharpe)
        # record results
        results['Iteration'].append(i+1)
        results['Sharpe Ratio'].append(sharpe)
        results['Weights'].append(weights)
        # check for sharpe ratio decrease
        if sharpe < recent_sharpe:
            counter += 1
            if counter >= patience:
                # print('There was 5 consecutive decreases in Sharpe Ratio, Terminating...')
                break
        else:
            recent_sharpe = sharpe
            counter = 0

    print('Total Iterations Completed: ' + str(max(results['Iteration'])))
    print('Optimal Weights: ' + str(weights))
    print('Sharpe Ratio: ' + str(sharpe))
    
    return results
```

---

## Portfolio Analysis

### 1. Before Optimization

```
%run /Users/Developer/OneDrive/Desktop/stock/main.ipynb
%load_ext autoreload
%autoreload 2
```

```
# define current portfolio & import data
stocks = ['AMZN', 'GOOGL', 'MSFT', 'NVDA', 'VOO']
weights = [0.1962, 0.2806, 0.1449, 0.0873, 0.2910]
data = import_data(stocks)
```

```
# create corrolation matrix
returns_data = pd.DataFrame({stock: data[stock]['Return'] for stock in data.keys()})
corr = returns_data.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(8, 6))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
plt.title("Stock Returns Correlation Matrix")
plt.show()
```

![image not available](1.PNG)


#### **Insights from the Correlation Matrix**
- **Highest Correlation**:
    - MSFT & VOO &rarr; 0.76
    - GOOGL & VOO &rarr; 0.69
- **Moderate Correlation**:
    - AMZN & GOOGL &rarr; 0.60
    - MSFT & GOOGL &rarr; 0.64
- **Lowest Correlation**:
    - NVDA & GOOGL &rarr; 0.50
    - NVDA & AMZN &rarr; 0.46

The correlation matrix reveals that MSFT, GOOGL, and VOO are highly correlated, indicating significant exposure to overall market movements. NVDA has the lowest correlations, providing diversification.

---

#### **Value at Risk (VaR)**

Value at Risk (VaR) estimates the maximum expected loss of a portfolio over a given period at a specific confidence level. VaR is defined as,

$VaR_a = \text{Percentile}_{1-\alpha}\text{(Portfolio Returns)}$

where:
- $\alpha$ = Confidence level (held at 95%)
- $\text{Portfolio Returns}$ = Weighted sum of stock returns

```
# compute VaR (value at risk) at confidence level 95%
def value_at_risk(data, weights, confidence_level=0.95):
    returns_data = pd.DataFrame({stock: data[stock]['Return'] for stock in data.keys()})
    returns_data.dropna(inplace=True)
    # compute portfolio daily returns
    portfolio_returns = np.dot(returns_data, weights)
    # compute VaR
    var = np.percentile(portfolio_returns, 100 * (1 - confidence_level))
    
    return var

var = value_at_risk(data, weights)
print('Portfolio VaR: ' + str(var))
```

**Output:**

```
Portfolio VaR: -0.022191281760184048
```

A VaR of -2.21% at a 95% confidence level tells me that under normal market conditions, my portfolio expects to lose no more than 2.22% in one day with a 95% probability. There is a 5% chance the losses could exceed this amount.

---

#### **Conditional Value at Risk (CVaR)**

To measure the expected loss in worst-case scenarios, we use Conditional Value at Risk (CVaR).

$CVaR_\alpha = E[R|R \leq VaR_\alpha]$

where:
- $VaR_\alpha$ = VaR at confidence level $\alpha$
- $E[R|R \leq VaR_\alpha]$ = Expected loss beyond the VaR threshold

```
# compute CVaR (conditional value at risk) at confidence level 95%
def conditional_value_at_risk(data, weights, confidence_level=0.95):
    returns_data = pd.DataFrame({stock: data[stock]['Return'] for stock in data.keys()})
    returns_data.dropna(inplace=True)
    portfolio_returns = np.dot(returns_data, weights)
    # compute VaR
    var = value_at_risk(data, weights)
    # compute CVaR
    cvar = portfolio_returns[portfolio_returns <= var].mean()
    return cvar

cvar = conditional_value_at_risk(data, weights)
print('Portfolio CVaR: ' + str(cvar))
```

**Output**: 

```
Portfolio CVaR: -0.0327431524600203
```

A CVaR of -3.27% at a 95% confidence level tell me that in the worst 5% cases, my portfolio's average daily loss is expected to be at least 3.27%.

---

```
# compute portfolio volatility
p_volatility = portfolio_volatility(data, weights)
print('Volatility: ' + str(p_volatility))

# compute expected portfolio return
p_return = portfolio_return(data, weights)
print('Expected Portfolio Return: ' + str(p_return))

# compute Sharpe Ratio
sharpe = sharpe_ratio(data, weights)
print('Share Ratio: ' + str(sharpe))
```

**Output**: 
```
Volatility: 0.2870430859720735
Expected Portfolio Return: 0.4098726354644012
Share Ratio: 1.288561381688832
```

#### **Before Optimization Portfolio Analysis**

We have calculated the following metrics for my portfolio before optmization:

| Metric | Value |
|-|-|
| VaR (95%) | -2.21\% |
| CVaR (95%) | -3.27\% |
| Volatility | 28.70\% |
| Return | 40.99\% |
| Sharpe Ratio | 1.29 |


---

#### **Exploring Portfolio Risk and Return**

Next, we generate 500 random portfolio weight combinations and evaluate their expected return, volatility, and Sharpe Ratio. My goal is to visualize the trade-off between risk and return and identify portfolios with the best risk-adjusted performance.

```
# generate n random weights
def generate_random_weights(n):
    rand_nums = np.random.rand(n)
    return rand_nums / sum(rand_nums)
```

```
# sample 500 random weights
weights_random = np.array([generate_random_weights(len(stocks)) for i in range(500)])
```

```
# calculate sharpe ratios
sharpe_ratios = {}
for j in range(len(weights_random)):
    sharpe = sharpe_ratio(data, weights_random[j])
    sharpe_ratios[j] = sharpe

# calculate expected portfolio returns
portfolio_returns = {}
for k in range(len(weights_random)):
    p_return = portfolio_return(data, weights_random[k])
    portfolio_returns[k] = p_return

# calculate volatility
volatilities = {}
for l in range(len(weights_random)):
    volatility = portfolio_volatility(data, weights_random[l])
    volatilities[l] = volatility

# stick everything in a dataframe
random_weights_df = pd.DataFrame({
    'Portfolio': list(portfolio_returns.keys()),
    'Expected Return': list(portfolio_returns.values()),
    'Volatility': list(volatilities.values()),
    'Sharpe Ratio': list(sharpe_ratios.values()),
})

random_weights_df.set_index('Portfolio', inplace=True)
```

```
# plot scatterplot (return vs risk)
sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.scatterplot(data=random_weights_df, 
                x="Expected Return", 
                y="Volatility",
                hue="Sharpe Ratio",
                palette="viridis",
                edgecolor="black",
                s=80,
                alpha=0.8
)

plt.title('Return vs Risk', fontsize=14, fontweight='bold')
plt.xlabel('Portfolio Volatility', fontsize=12)
plt.ylabel('Expected Portfolio Return', fontsize=12)
plt.legend(title="Sharpe Ratio", fontsize=10, title_fontsize=12, loc='best')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
```

![image not available](2.PNG)

#### **Observations from the Graph**:

1. **Positive Correlation**: High portfolio volatility generally leads to higher expected returns
2. **Sharpe Ratio**: The scatterplot indicates higher sharpe ratios (i.e better risk-return tradeoffs) at the upper edge
3. **Diversification**: Some portfolios have similar returns at lower risk, which suggests optimal weight distributions
4. **Best Portfolios**: The portfolios with the highest Sharpe Ratios (yellow ones) are the most efficient, maximizing return per unit of risk

The graph shows that riskier portfolios have higher returns. The curve's shape suggests that some portfolios are better than others at balancing risk and return. The color gradient helps to pick out the most efficient portfolios, which offer the highest return for the amount of risk taken. Overall, this shows why it is important to get the right mix of stocks, a diversified portfolio can increase returns without undue risk.

---

### 2. After Optimization

#### **Optimizing the Sharpe Ratio**

I will now use gradient ascent to maximize the Sharpe Ratio by manipulating the portfolio's weights. The algorithm will ieratively move in a direction towards improving risk-adjust return and will stop when it reaches an optimal point. 

```
weights = generate_random_weights(5)
optimize = portfolio_optimization(data, weights, alpha=0.5)
```

**Output**:

```
Total Iterations Completed: 52
Optimal Weights: [0.30054491 0.00997087 0.13299555 0.54651781 0.00997087]
Sharpe Ratio: 1.3248368626574984
```

#### **Computed Optimal Portfolio**

| Stock | Weight |
|-|-|
| AMZN |  30.05% |
| GOOGL | 1.00% |
| MSFT | 13.30% |
| NVDA | 54.65% |
| VOO | 1.00% |

The algorithm converged in 52 iterations with a 1.3248 Sharpe Ratio, an improvement over my initial portfolio. The optimal weights show a significant shift, with NVDA having the highest allocation (54.65%), while GOOGL & VOO are minimized (1%). 

---

```
plt.figure(figsize=(8, 4))
plt.plot(optimize['Iteration'], optimize['Sharpe Ratio'], marker='o', color='blue', markersize='5', label='Sharpe Ratio')

plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Sharpe Ratio', fontsize=12)
plt.title('Sharpe Ratio Convergence Over Iterations', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()
```

![image not available](3.PNG)

#### **Convergence Over Iterations**

The graph above shows the Sharpe Ratio's convergence over iterations. The Sharpe Ratio starts low (~1.17) but then sharply climbs within the first 10 iterations, converging at 1.32. This shows the algorithm produces a fairly smooth convergence quickly and efficiently.

---

```
# define optimized weights
optimized_weights = optimize['Weights'][-1]

# compute new VaR, CVaR, volatility, expected return, and sharpe ratio
optimized_var = value_at_risk(data, optimized_weights)
optimized_cvar = conditional_value_at_risk(data, optimized_weights)
optimized_volatility = portfolio_volatility(data, optimized_weights)
optimized_return = portfolio_return(data, optimized_weights)
optimized_sharpe = sharpe_ratio(data, optimized_weights)

print('New VaR: ' + str(optimized_var))
print('New CVaR: ' + str(optimized_cvar))
print('New Volatility: ' + str(optimized_volatility))
print('New Expected Portfolio Return: ' + str(optimized_return))
print('New Sharpe Ratio: ' + str(optimized_sharpe))
```

**Output**:

```
New VaR: -0.03213355925697557
New CVaR: -0.0467360717002777
New Volatility: 0.3318032130214905
New Expected Portfolio Return: 0.47958512775906914
New Sharpe Ratio: 1.3248368626574984
```

#### **Final Analysis: Before vs After Optimization**

| Metric | Before Optimization | After Optimization | Difference
|-|-|-|-|
| VaR | -2.21% | -3.21% | -1.00 |
| CVaR | -3.27% | -4.67% | -1.40 |
| Volatility | 28.70% | 33.18% | 4.40 |
| Return | 40.99% | 47.96% | 6.97 |
| Sharpe Ratio | 1.29 | 1.32 | 0.03 |

**1. Increased Return (+6.97%)**
- Expected return increased from 40.99% to 47.96%, showing the algorithm allocated more weight to higher-return assets.
- This suggests the algorithm shifted towards more aggresive stocks, prioritizing higher gains.

**2. Higher Volatility (+4.40)** 
- Portfolio volatility increased from 28.70% to 33.18%, showing the optimizer favored higher-risk stocks.
- This is to be expected, as we saw earlier that higher returns come with increased volatility.

**3. Greater Downside Risk (VaR and CVaR Decreased)**
- VaR decreased from -2.21% to -3.21%, indicating the optimized portfolio has higher potential daily loss at the 95% confidence level.
- CVaR also decreased from -3.27% to -4.67%, indicating average losses in extreme scenarios are now larger.
- This suggests the optimizer shifted weight towards more volatile stocks that carry higher downside risk.

**4. Improvement in Sharpe Ratio (+0.03)**
- The Sharpe Ratio improved from 1.29 to 1.32, indicating the optimized portfolio earns more return per unit of risk.
- This shows the optimizer did its job and successfully maximized the Sharpe Ratio.

---

## Conclusion

The optimization successfully increased expected returns while improving risk-adjusted performance (Sharpe Ratio). However, the increased downside risk indicates that this portfolio is better suited for more aggresive investors willing to accept higher volatility.

**Key Findings:**
- The optimized portfolio achieved a higher return (+6.79%), increasing from 40.99% to 47.96%
- Volatility increased (+4.40%), indicating a shift toward riskier stocks to get higher returns
- Downside risk got worse, with VaR dropping from -2.21% to -3.21% and CVaR from -3.27% to -4.67%
- The Sharpe Ratio improved from 1.29 to 1.32, showing the optimizer succesfully did its job

For a more balanced or conservative approach, further implementations could include risk constraints to limit volatility and extreme losses while still optimizing returns.

**Future Enhancements:**
- Implement Additional Constraints (e.g., capping volatility)
- Explore Other Optimization Techniques 
- Conduct Stress Testing to evaluate the portfolio’s performance under historical market crises.