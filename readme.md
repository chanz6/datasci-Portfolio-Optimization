<a id="readme-top"></a>

<!-- PROJECT LOGO -->

  <h3 align="center">Stock Portfolio Optimizer</h3>

  <p align="center">
    An optimization model to maximize the Sharpe Ratio of a stock portfolio
    <br />
    <a href="https://github.com/chanz6"><strong>More Personal Projects »</strong></a>
    <br />
  </p>
</div>

[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

This project aims to maximize the Sharpe ratio via gradient ascent, working towards creating smarter, fact-based investment strategies while balancing return and risk. By using quantitative analysis and machine learning, it optimizes portfolio allocations for the most efficient risk-adjusted performance. The process follows a structured and intuitive workflow:

- **Data Collection** – Gathering stock information from Yahoo Finance to capture market trends.
- **Exploratory Analysis** – Identifying trends, relationships, and key insights within the data.
- **Risk Analysis** – Assessing portfolio volatility, downside risk, and drawdowns to maintain stability.
- **Optimization** – Iteratively adjusting portfolio weights using gradient ascent to enhance performance.
- **Visualization** – Presenting results through charts and reports for clear and actionable insights.

Built with Python and Jupyter, this project incorporates libraries like NumPy, Pandas, and Matplotlib. It’s a great project for anyone looking to explore portfolio optimization, risk management, and data-driven investment strategies through real-world application

### Built With

* ![Python][Python]
* ![Pandas][Pandas]
* ![Numpy][Numpy]
* ![Matplotlib][Matplotlib]
* ![Seaborn][Seaborn]
* ![YFinance][YFinance]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

For the model to function, the following must be installed first:
* jupyter notebook
  ```sh
  pip install jupyter
  ```
* nbformat
  ```sh
  pip install nbformat
  ```
* pandas
  ```sh
  pip install pandas
  ```
* numpy
  ```sh
  pip install numpy
  ```
* matplotlib
  ```sh
  pip install matplotlib
  ```
* seaborn
  ```sh
  pip install seaborn
  ```
* yfinance
  ```sh
  pip install yfinance
  ```

### Installation

This repo can be cloned using the following command:
   ```sh
   git clone https://github.com/chanz6/datasci-Portfolio-Optimization.git
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

1. Navigate to the `usage.ipynb` file
2. Input your portfolio information
3. Run cells below to analyze your portfolio and optimize!

### Example Usage

Suppose this is the stock portfolio we wish to optimize:

| Stock | Name | Weight |
|-|-|-|
| NVDA | Nvidia | 10% |
| MA | Mastercard | 70% |
| NFLX | Netflix | 5% |
| CMG | Chipotle | 7% |
| MNST | Monster | 8% |

Then in `usage.ipynb`, we input the portfolio information and run the cell

```
# enter portfolio information here
stocks = ['NVDA', 'MA', 'NFLX', 'CMG', 'MNST']
weights = [0.10, 0.70, 0.05, 0.07, 0.08]
```

Run the `initial_analysis` cell to analyze your current portfolio

![Correlation matrix](/images/4.PNG)

| | Metric | Value |
|-|-|-|
|0|VaR|-2.36%|
|1|CVaR|-3.62%|
|2|Volatility|24.66%|
|3|Return|32.90%|
|4|Sharpe Ratio|1.17|

Run the `final_analysis` cell to compute our optimized portfolio and metrics

```
Total Iterations Completed: 16
Sharpe Ratio: 1.509349448183119
Computed Optimal Portfolio:
NVDA    30.85%
MA      10.84%
NFLX    20.38%
CMG     19.35%
MNST    18.57%
dtype: object
```

![Convergence](/images/5.PNG)

| | Metric | Value |
|-|-|-|
|0|VaR|-2.58%|
|1|CVaR|-3.91%|
|2|Volatility|27.29%|
|3|Return|45.19%|
|4|Sharpe Ratio|1.51|

<p align="right">(<a href="#readme-top">back to top</a>)</p>

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=0077B5
[linkedin-url]: https://www.linkedin.com/in/zachary-chann/
[product-screenshot]: images/screenshot.png
[Python]: https://img.shields.io/badge/python-000000?style=for-the-badge&logo=python&logoColor=blue
[Pandas]: https://img.shields.io/badge/Pandas-000bff?style=for-the-badge&logo=pandas&logoColor=purple
[Numpy]: https://img.shields.io/badge/NumPy-ad526f?style=for-the-badge&logo=NumPy&logoColor=blue
[Matplotlib]: https://img.shields.io/badge/Matplotlib-DD0031?style=for-the-badge&logo=matplotlib&logoColor=white
[Seaborn]: https://img.shields.io/badge/Seaborn-4A4A55?style=for-the-badge&logo=seaborn&logoColor=FF3E00
[Yfinance]: https://img.shields.io/badge/yfinance-563D7C?style=for-the-badge&logo=&logoColor=white
