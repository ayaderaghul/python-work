# wheel_prophet_bs_full.py

import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from scipy.stats import norm
import math
from datetime import timedelta, datetime
import os
import random

# ------------------------------
# CONFIG
# ------------------------------
class Config:
    ticker = "AAPL"
    lookback_days = 365
    forecast_days = 30
    r = 0.05  # risk-free rate
    dte_put = 30   # days to expiry for puts
    dte_call = 30  # days to expiry for calls
    target_delta_put = -0.15
    target_delta_call = 0.2
    capital = 100000
    cut_loss_pct = 0.7
    min_premium_yield_monthly = 0.02
    outputs_folder = "outputs"

config = Config()
os.makedirs(config.outputs_folder, exist_ok=True)

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def fetch_stock_history(ticker, period="1y"):
    data = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    data = data.reset_index()
    data = data[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
    return data

def fallback_hv_sigma(prices):
    if len(prices) < 2:
        return 0.2  # default guess if not enough data
    returns = np.log(prices / prices.shift(1)).dropna()
    if len(returns) == 0:
        return 0.2
    sigma = returns.std() * np.sqrt(252)
    if np.isnan(sigma):
        return 0.2
    # if sigma is None or np.isnan(sigma) or sigma <= 0:
    #     sigma = 0.2  # fallback default TODO
    return sigma

def run_prophet_forecast(prices, periods=30):
    # Remove timezone
    prices['ds'] = prices['ds'].dt.tz_localize(None)
    
    m = Prophet(daily_seasonality=True)
    m.fit(prices)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return forecast[['ds','yhat','yhat_lower','yhat_upper']]

# ------------------------------
# BLACK-SCHOLES PRICING & GREEKS
# ------------------------------
def bs_price(S, K, r, sigma, T, option_type='call', q=0.0):
    d1 = (math.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    if option_type=='call':
        price = S*math.exp(-q*T)*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
        delta = math.exp(-q*T)*norm.cdf(d1)
    else:
        price = K*math.exp(-r*T)*norm.cdf(-d2) - S*math.exp(-q*T)*norm.cdf(-d1)
        delta = math.exp(-q*T)*(norm.cdf(d1)-1)
    theta = (-S*sigma*norm.pdf(d1)*math.exp(-q*T)/(2*math.sqrt(T)) -
             r*K*math.exp(-r*T)*norm.cdf(d2 if option_type=='call' else -d2)) / 365
    return price, delta, theta

# ------------------------------
# PICK STRIKE BY TARGET DELTA
# ------------------------------
def pick_strike_by_delta(S, sigma, T, target_delta, option_type='call'):
    if option_type == 'call':
        u = norm.ppf(target_delta)
    else:  # put
        u = norm.ppf(target_delta + 1.0)
    K = S * math.exp(- u * sigma * math.sqrt(T) + 0.5 * sigma**2 * T)
    return K

def choose_nearest_strike(K_cont, available_strikes):
    if not available_strikes:
        return K_cont
    available_strikes = sorted(available_strikes)
    nearest = min(available_strikes, key=lambda x: abs(x - K_cont))
    return nearest

# ------------------------------
# FETCH OPTIONS WITH GREEKS
# ------------------------------
def fetch_options_with_greeks(ticker, expirations, S, r, fallback_sigma):
    all_options = []
    tk = yf.Ticker(ticker)
    for exp in expirations:
        chain = tk.option_chain(exp)
        for df, opt_type in [(chain.calls,'call'), (chain.puts,'put')]:
            available_strikes = df['strike'].tolist()
            for _, row in df.iterrows():
                K = row['strike']
                T = (pd.to_datetime(exp) - datetime.today()).days / 365.0
                sigma = row['impliedVolatility']
                if sigma is None or sigma<=0: sigma = fallback_sigma
                price, delta, theta = bs_price(S, K, r, sigma, T, opt_type)
                all_options.append({'expiry':exp, 'type':opt_type, 'strike':K,
                                    'bs_price':price, 'bs_delta':delta,
                                    'bs_theta':theta, 'sigma_used':sigma})
    return pd.DataFrame(all_options)

# ------------------------------
# MDP SIMULATOR LOOP
# ------------------------------
def phys_assign_prob(S, K, T, sigma, mu):
    z = (math.log(K/S) - (mu - 0.5*sigma**2)*T) / (sigma * math.sqrt(T))
    return norm.cdf(z)  # put assign probability

def simulate_wheel(dates, price_series, forecast_series, config):
    cash = config.capital
    positions = []
    stock_lots = []
    hist = []

    for t_idx, today in enumerate(dates):
        S = price_series[t_idx]

        # handle expiries
        expiring = [p for p in positions if p['expiry'] == today]
        for p in expiring:
            # determine assignment probability
            mu = math.log(forecast_series.iloc[min(t_idx+config.forecast_days,len(forecast_series)-1)]['yhat']/S) / (config.forecast_days/365.0)
            T = max((p['expiry'] - today).days / 365.0, 1/252)
            if p['type']=='short_put':
                p_assign = phys_assign_prob(S, p['K'], T, p['sigma_used'], mu)
                if random.random() < p_assign:
                    shares = 100 * p['contracts']
                    stock_lots.append({'shares':shares, 'cost':p['K']})
                    cash -= shares * p['K']
                positions.remove(p)
            elif p['type']=='covered_call':
                p_assign = 1 - phys_assign_prob(S, p['K'], T, p['sigma_used'], mu)
                if random.random() < p_assign:
                    shares_to_sell = 100 * p['contracts']
                    # FIFO sell
                    shares_remaining = shares_to_sell
                    new_lots = []
                    for lot in stock_lots:
                        if shares_remaining >= lot['shares']:
                            cash += lot['shares']*p['K']
                            shares_remaining -= lot['shares']
                        else:
                            cash += shares_remaining*p['K']
                            lot['shares'] -= shares_remaining
                            new_lots.append(lot)
                            shares_remaining=0
                    stock_lots = new_lots
                positions.remove(p)
        # policy decision
        if cash >= S*100:  # enough to sell CSP
            T = config.dte_put/365.0
            sigma = fallback_hv_sigma(price_series[:t_idx+1])
            K_cont = pick_strike_by_delta(S, sigma, T, config.target_delta_put, 'put')
            K = round(K_cont)  # could replace with nearest strike from chain
            price, delta, theta = bs_price(S, K, config.r, sigma, T, 'put')
            cash += price*100
            positions.append({'type':'short_put', 'K':K, 'premium':price, 'expiry':today + timedelta(days=config.dte_put), 'contracts':1, 'sigma_used':sigma})
        elif stock_lots:
            T = config.dte_call/365.0
            sigma = fallback_hv_sigma(price_series[:t_idx+1])
            K_cont = pick_strike_by_delta(S, sigma, T, config.target_delta_call, 'call')
            K = round(K_cont)
            price, delta, theta = bs_price(S, K, config.r, sigma, T, 'call')
            positions.append({'type':'covered_call', 'K':K, 'premium':price, 'expiry':today + timedelta(days=config.dte_call), 'contracts':1, 'sigma_used':sigma})

        # NAV record
        nav = cash + sum([lot['shares']*S for lot in stock_lots])
        hist.append({'date':today, 'nav':nav, 'cash':cash, 'stock_value':sum([lot['shares']*S for lot in stock_lots])})

    return pd.DataFrame(hist)

# ------------------------------
# MAIN SCRIPT
# ------------------------------
if __name__=="__main__":
    prices = fetch_stock_history(config.ticker, period="1y")
    prices.to_csv(f"{config.outputs_folder}/{config.ticker}_prices.csv", index=False)
    forecast = run_prophet_forecast(prices, periods=config.forecast_days)
    forecast.to_csv(f"{config.outputs_folder}/{config.ticker}_forecast.csv", index=False)

    S_current = prices['y'].iloc[-1]
    fallback_sigma = fallback_hv_sigma(prices['y'])
    
    tk = yf.Ticker(config.ticker)
    expirations = tk.options[:3]  # first 3 expiries
    options_df = fetch_options_with_greeks(config.ticker, expirations, S_current, config.r, fallback_sigma)
    options_df.to_csv(f"{config.outputs_folder}/{config.ticker}_options_bs.csv", index=False)

    # Run MDP simulator
    hist_df = simulate_wheel(prices['ds'], prices['y'], forecast, config)
    hist_df.to_csv(f"{config.outputs_folder}/{config.ticker}_mdp_sim.csv", index=False)
    print("Simulation complete! CSVs saved in 'outputs/' folder.")
