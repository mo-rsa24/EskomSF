import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_season(month: int) -> str:
    """Season mapping for Southern Hemisphere (South Africa)."""
    if month in [12, 1, 2]:
        return "Summer"
    elif month in [3, 4, 5]:
        return "Autumn"
    elif month in [6, 7, 8]:
        return "Winter"
    elif month in [9, 10, 11]:
        return "Spring"
    return "Unknown"


def engineer_features(
    df: pd.DataFrame,
    target_col: str,
    sa_holidays: set = None,
    lag_months: list = [12, 24, 36, 48, 60],
    add_calendar: bool = True,
    use_extended_calendar_features: bool = True,
    drop_na: bool = True
) -> pd.DataFrame:
    """
    Generate lagged and calendar features for time-series forecasting.

    Parameters:
        df: DataFrame with datetime index
        target_col: Column to forecast (e.g., PeakConsumption)
        sa_holidays: Set of datetime.date objects for public holidays
        lag_hours: List of lag intervals to create
        add_calendar: Add basic calendar features (dow, weekend, holiday)
        use_extended_calendar_features: Add season, month, quarter, year
        drop_na: Whether to drop rows after lagging

    Returns:
        Feature-enhanced DataFrame
    """
    df = df.copy()
    if sa_holidays is None:
        sa_holidays = set()

    # 📅 Calendar Features
    if add_calendar:
        df['dayofweek'] = df.index.dayofweek
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['holiday'] = pd.Series(df.index.date, index=df.index).map(lambda d: int(d in sa_holidays))
        df['day'] = df.index.day
        df['hour'] = df.index.hour  # Will be 0 for monthly data, but safe for hourly

    if use_extended_calendar_features:
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['Year'] = df.index.year
        df['Season'] = df['Month'].map(get_season)
        df = pd.get_dummies(df, columns=['Season'], prefix='Season')

    # 🔁 Lag Features (only if enough rows)
    if len(df) > max(lag_months):
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        for lag in lag_months:
            df[f"{target_col}_lag{lag}"] = df[target_col].shift(periods=lag)
    else:
        logger.warning(
            f"⚠️ Skipping lag features: only {len(df)} rows but max lag requested is {max(lag_months)}."
        )
    # 🧹 Final cleanup
    if drop_na:
        df.dropna(inplace=True)

    return df



def grouped_engineer_features(
    df: pd.DataFrame,
    target_col: str,
    sa_holidays: set = None,
    lag_months: list = [12, 24, 36, 48, 60],
    add_calendar: bool = True,
    use_extended_calendar_features: bool = True,
    drop_na: bool = True,
    group_keys: list = ['CustomerID', 'PodID']
) -> pd.DataFrame:
    """
    Apply feature engineering grouped by CustomerID and PodID.
    """

    all_groups = []

    for group_key, group_df in df.groupby(group_keys):
        group_df = group_df.sort_index()

        engineered = engineer_features(
            df=group_df,
            target_col=target_col,
            sa_holidays=sa_holidays,
            lag_months=lag_months,
            add_calendar=add_calendar,
            use_extended_calendar_features=use_extended_calendar_features,
            drop_na=drop_na
        )

        for k, col in zip(group_keys, group_key if isinstance(group_key, tuple) else [group_key]):
            engineered[k] = col

        all_groups.append(engineered)

    return pd.concat(all_groups).sort_index()

