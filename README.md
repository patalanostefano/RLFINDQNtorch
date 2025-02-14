# RLFINDQNtorch: DQ-Learning Trading System that leverages momentum and technical indicators

## Overview
This project implements a Deep Q-Learning (DQL) approach for establishing long-term trading policies on single stocks and indices. The system uses historical market data and various technical indicators to make trading decisions, with a special focus on trend following and feature extraction.

## Key Features
- Custom trading environment based on OpenAI Gym
- Support for real-time data using yfinance
- Multiple model architectures including DQN and Encoder-Decoder
- Custom trend following approach
- Integration of technical indicators (MACD, RSI, BB, VOLUME, EMA)
- Flexible state representation with OHLCV data

## System Architecture

### Trading Environment
- Custom implementation inheriting from `gym.Env`
- Single share trading logic (one stock at a time)
- N-step ahead reward calculation
- Support for data normalization and indicator calculation
- Trend strength calculation using Aaron indicator

### Model Architectures

1. **Simple DQN**
   - Basic DQN architecture
   - Configurable width parameters for OHLCV and indicator representations
   
2. **Encoder-Decoder (SimpleMLP)**
   - Enhanced state representation through encoding layer
   - Improved feature extraction capabilities

3. **Custom Encoder-Decoder with Trend Following**
   - Specialized architecture for trend following
   - Custom memory sampling based on trend duration

### Training Process
1. Episode-based training with environment reset
2. Action selection and environment stepping
3. Transition storage in replay memory
4. Periodic target network updates

### Optimization Process
1. Batch sampling from replay memory
2. Q-value computation using policy network
3. Expected Q-value estimation using target network
4. Huber loss calculation and backpropagation
5. Policy network weight updates

## Performance Features

### Trend Following Approach
- Trend duration calculation during data preparation
- Selective memory sampling based on trend strength
- Improved performance for mid-duration trends

### Technical Indicators
- Integration of multiple technical indicators
- Performance stabilization across different assets
- Enhanced short-term fluctuation capture for index-based assets

## Results

The system has been tested on various assets including:
- S&P 500
- NASDAQ
- GOLD

Key findings:
Custom trend following approach shows superior performance:
- Indicator inclusion stabilizes performance across different assets
- Index-based assets benefit from technical indicators
- Commodities perform better with trend-following strategies leveraging momentum



## Future Extensions

### Multi-Trading Environment
- Multiple buy/sell actions
- Complex reward function for diverse outcomes
- Balance between short-term and long-term rewards

### Portfolio Optimization
- Multi-stock environment support
- Weighted multi-reward structure
- Portfolio diversification and risk management

## Dependencies
- OpenAI Gym
- yfinance

## References
1. Learning Financial Asset-Specific Trading Rules via Deep Reinforcement Learning
2. [Gym Trading Environment Documentation](https://gym-trading-env.readthedocs.io/en/latest/index.html)


## Author
Stefano Patalano
