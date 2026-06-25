# -------------
from abc import ABC, abstractmethod
from typing import List, Tuple
# -------------
import numpy as np
import pandas as pd
# -------------

class DataProcessor:
    def __init__\
    (
        self, 
        lags: int = 5
    ):
        self.lags = lags

    def generate_features\
    (
        self, 
        df: pd.DataFrame
    ) -> pd.DataFrame:
        # -------------
        data = df.copy()
        # -------------
        
        # -------------
        data["returns"] = np.log\
            (
                data["close"] / data["close"].shift(1)
            )
        # -------------
        
        # -------------
        for lag in range(1, self.lags + 1):
            data[f"lag_return_{lag}"] = data["returns"].shift(lag)
            data[f"lag_price_{lag}"] = data["close"].shift(lag)
        # -------------
        
        return data.dropna()

class BaseStrategy(ABC):
    @abstractmethod
    def train\
    (
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> None:
        """
            Train predicted model
        """
        pass

    @abstractmethod
    def generate_signals\
    (
        self, 
        X: pd.DataFrame
    ) -> np.ndarray:
        """
            Generate trading signals: 
            (*) +01 (Buy) 
            (*) -01 (Sell) 
            (*) *00  (Neutral)
        """
        pass

class MLTimeSeriesStrategy(BaseStrategy):
    def __init__\
    (
        self, 
        model
    ):
        # -------------
        self.model = model
        self.feature_cols: List[str] = []
        # -------------
        
    def train\
    (
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> None:
        # -------------
        self.feature_cols = X.columns.tolist()
        # -------------
        
        # -------------
        self.model.fit(X, y)
        # -------------
        
    def generate_signals\
    (
        self, 
        X: pd.DataFrame
    ) -> np.ndarray:
        # -------------
        if not self.feature_cols:
            raise ValueError\
            (
                "Model should be trained before generating signals"
            )
        # -------------
        
        # -------------
        predictions = self.model.predict(X[self.feature_cols])
        # -------------
        
        # -------------
        signals = np.where(predictions > 0, 1, -1)
        # -------------
        
        return signals

class WalkForwardBacktester:
    def __init__\
    (
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy,
        initial_capital: float = 100000.0,
        execution_cost: float = 0.0005,
    ):
        # -------------
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.execution_cost = execution_cost
        # -------------
        
    def run_backtest\
    (
        self, 
        train_size: int
    ) -> pd.DataFrame:        
        # -------------
        close_prices = self.data["close"].values
        returns = self.data["returns"].values
        # -------------
        
        # -------------
        X = self.data.filter(like="lag_")
        y = self.data["returns"].shift(-1).fillna(0)
        # -------------
        
        # -------------
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        # -------------
        
        # -------------
        self.strategy.train(X_train, y_train)
        # -------------
        
        # -------------
        test_signals = self.strategy.generate_signals(X_test)
        # -------------
        
        # -------------
        results = pd.DataFrame(index=X_test.index)
        results["actual_returns"] = returns[train_size:]
        results["signal"] = test_signals
        # -------------
        
        # -------------
        results["position_changes"] =\
        (
            results["signal"].diff().fillna(0).abs()
        )
        
        results["strategy_returns"] =\
        (
            results["signal"] * results["actual_returns"]
        ) - (results["position_changes"] * self.execution_cost)
        # -------------
        
        # -------------
        results["cum_market_returns"] =\
        (
            1 + results["actual_returns"]
        )\
        .cumprod() * self.initial_capital
        
        results["cum_strategy_returns"] =\
        (
            1 + results["strategy_returns"]
        )\
        .cumprod() * self.initial_capital
        # -------------
        
        return results

class PerformanceEvaluator:
    @staticmethod
    def calculate_metrics\
    (
        results: pd.DataFrame, 
        trading_days: int = 252
    ) -> dict:
        # -------------
        strat_returns = results["strategy_returns"]
        # -------------
        
        # -------------
        total_return =\
        (
            results["cum_strategy_returns"].iloc[-1]
            / results["cum_strategy_returns"].iloc[0]
        ) - 1
        
        num_years = len(results) / trading_days
        annualized_return = (1 + total_return) ** (1 / num_years) - 1
        # -------------
        
        # -------------
        annualized_vol = strat_returns.std() * np.sqrt(trading_days)
        # -------------
        
        # -------------
        sharpe_ratio =\
        (
            annualized_return / annualized_vol if annualized_vol != 0 else 0
        )
        # -------------
        
        # -------------
        equity_curve = results["cum_strategy_returns"]
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()

        result=\
        (
            {
                "Annualized Return": f"{annualized_return:.02%}",
                "Annualized Volatility": f"{annualized_vol:.02%}",
                "Sharpe Ratio": f"{sharpe_ratio:.02f}",
                "Max Drawdown": f"{max_drawdown:.02%}",
            }
        )

        return result

if __name__ == "__main__":
    # -------------
    np.random.seed(42)
    dates = pd.date_range\
                (
                    start="2026-01-01",
                    periods=1000, 
                    freq="h"
                )
                
    mock_prices = 100 + np.random.randn(1000).cumsum()
    raw_data = pd.DataFrame\
                (
                    {
                        "close": mock_prices
                    }, 
                    index=dates
                )
    # -------------
    print("------------------")
    print("raw_data:")
    print(raw_data)
    print("------------------")
    
    # -------------
    processor = DataProcessor(lags=5)
    processed_df = processor.generate_features(raw_data)
    # -------------
    
    # -------------
    from sklearn.ensemble import GradientBoostingRegressor
    # -------------
    
    # -------------
    model_instance = GradientBoostingRegressor\
    (
        n_estimators=50, 
        max_depth=3, 
        random_state=42
    )
    strategy = MLTimeSeriesStrategy\
    (
        model=model_instance
    )
    # -------------

    # -------------
    train_split_index = int\
                        (
                            len(processed_df) * 0.7
                        )
    backtester = WalkForwardBacktester\
    (
        data=processed_df, 
        strategy=strategy, 
        execution_cost=0.0002
    )
    backtest_results = backtester.run_backtest\
    (
        train_size=train_split_index
    )
    # -------------
    
    # -------------
    metrics = PerformanceEvaluator.calculate_metrics\
    (
        backtest_results,
        trading_days=252 * 24
    )
    # -------------

    # -------------
    print("")
    print(" Back Test Output ==================")    
    print("")

    for item in metrics.items():
        (metric_name, value) = item
        print(f"    {metric_name}:{value}")

    print("")
    print(" ===================================")
    print("")
    # -------------

