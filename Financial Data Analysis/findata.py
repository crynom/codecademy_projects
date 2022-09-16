import numpy as np
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import datetime as dt

def return_portfolios(expected_returns, cov_matrix):
    port_returns = []
    port_volatility = []
    stock_weights = []
    
    selected = (expected_returns.axes)[0]
    
    num_assets = len(selected) 
    num_portfolios = 5000
    
    for single_portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns = np.dot(weights, expected_returns)
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        port_returns.append(returns)
        port_volatility.append(volatility)
        stock_weights.append(weights)
    
    portfolio = {'Returns': port_returns,
                 'Volatility': port_volatility}
    
    for counter,symbol in enumerate(selected):
        portfolio[symbol +' Weight'] = [Weight[counter] for Weight in stock_weights]
    
    df = pd.DataFrame(portfolio)
    
    column_order = ['Returns', 'Volatility'] + [stock+' Weight' for stock in selected]
    
    df = df[column_order]
   
    return df
  
def optimal_portfolio(returns):
    n = returns.shape[1]
    returns = np.transpose(returns.to_numpy())

    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x']
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    ptwt = [np.asarray(p) for p in portfolios]
    
    return np.asarray(wt), returns, risks, ptwt
symbols = ['FB','AMD','AAPL','JBL','TSM']
end = dt.date.today()
start = end - dt.timedelta(days = 365)
stocks = web.get_data_yahoo(symbols, start, end)
adjc = stocks['Adj Close']
print(adjc.head(),end,start)
adjc.plot()
plt.title('Price Over Time')
plt.ylabel('Price')
plt.xlabel('Date')
print(adjc['AMD'].mean())
sror = adjc.pct_change()

fig = plt.figure(figsize=(15,15))
ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325)


ax1.plot(sror['FB'])
ax1.set_title('FB Simple Returns')
ax2.plot(sror['AMD'])
ax2.set_title('AMD Simple Returns')
ax3.plot(sror['AAPL'])
ax3.set_title('AAPL Simple Returns')
ax4.plot(sror['JBL'])
ax4.set_title('JBL Simple Returns')
ax5.plot(sror['TSM'])
ax5.set_title('TSM Simple Returns')
#mean variance portfolio optimization
selected = list(adjc)

daily_ret = adjc[selected].pct_change()
expected_ret = daily_ret.mean()
daily_cov = daily_ret.cov()

df = return_portfolios(expected_ret, daily_cov)
df = df.iloc[1:,:]

weights, returns, risks, ptwt = optimal_portfolio(daily_ret[1:])


pt = 1
print('Weights in order of call: ',ptwt[pt],'\nReturns: ', returns[pt],'\nRisk: ', risks[pt], '\n', weights)
df.plot.scatter(x = 'Volatility', y = 'Returns')
plt.plot(risks,returns,"g--")
plt.xlabel('Returns', fontsize = 14)
plt.ylabel('Volatility', fontsize = 14)
plt.title('Efficient Frontier', fontsize = 16)
print(ptwt)
plt.show()