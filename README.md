# Stochastic Sports Asset Pricing and Portfolio Optimization Engine

**Author: Ayush Parida

Focus: Computational Finance, Modern Portfolio Theory, Mixed Integer Linear Programming

 Overview

This project looks at player performance and team building in Fantasy Premier League the same way a quant fund looks at financial markets. Most casual players pick their teams based on emotion, lucky goals from last week, or whoever is popular on social media.

This engine cuts through that noise. It pulls live match stats directly from the Premier League API and models future performance using math instead of gut feeling.

 How the Engine Works

 10,000 Run Monte Carlo Simulations:Football is full of random bounces and luck. Instead of just guessing what a player will score next, the model runs 10,000 match simulations for every active player using a Poisson distribution. This gives us both their expected points and their downside risk.
Risk Adjusted Value with the Sharpe Ratio:** In finance, you never look at returns without looking at risk. A player who scores 6 points every single match is safer than a player who scores 15 points once and then blanks for a month. The model uses the Sharpe Ratio to rank players by how reliable their output is compared to their volatility.
Optimal Squad Solver: Finding the best 15 player squad under a strict 100 million budget cap with real world rules (2 goalkeepers, 5 defenders, 5 midfielders, 3 forwards, and at most 3 players from any single club) is a complex math problem. The script uses Mixed Integer Linear Programming to test millions of combinations and output the single best squad mathematically possible.
Testing Retail Herd Mentality: The model runs a regression comparing how many people own a player against how efficient that player actually is. This highlights market inefficiencies where popular players are overpriced and quiet, consistent players are heavily undervalued.

 Files and Live Links

Live web app https://fpl-stochastic-quant-engine-9uwtuucayumrg6vwdefzqw.streamlit.app/

* app.py: The full code running the live interactive dashboard
* Premier_League_Stochastic_Quant_Model.xlsx: The complete spreadsheet with raw stats, risk scores, and optimal picks of he players
