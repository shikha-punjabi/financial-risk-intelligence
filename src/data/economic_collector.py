import fredapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import time
from config.settings import Config

logger = logging.getLogger(__name__)

class EconomicDataCollector:
    def __init__(self, api_key: str):
        self.fred = fredapi.Fred(api_key=api_key)
        self.key_indicators = {
            # Growth indicators
            'GDP': 'GDP',
            'GDP_growth': 'A191RL1Q225SBEA',
            'industrial_production': 'INDPRO',
            # Employment
            'unemployment_rate': 'UNRATE',
            'nonfarm_payrolls': 'PAYEMS',
            'labor_participation': 'CIVPART',
            'jobless_claims': 'ICSA',
            # Inflation
            'cpi': 'CPIAUCSL',
            'core_cpi': 'CPILFESL',
            'pce': 'PCE',
            'core_pce': 'PCEPILFE',
            # Interest rates
            'fed_funds_rate': 'DFF',
            'treasury_10y': 'DGS10',
            'treasury_2y': 'DGS2',
            'treasury_3m': 'DGS3MO',
            # Consumer
            'consumer_sentiment': 'UMCSENT',
            'retail_sales': 'RSAFS',
            'personal_income': 'PI',
            'consumer_credit': 'TOTLL',
            # Housing
            'housing_starts': 'HOUST',
            'existing_home_sales': 'EXHOSLUSM495S',
            'home_price_index': 'CSUSHPISA',
            # Market indicators
            'vix': 'VIXCLS',
            'ted_spread': 'TEDRATE',
            'credit_spread': 'BAMLH0A0HYM2',
            # Money supply
            'money_supply_m2': 'M2SL',
            'commercial_loans': 'TOTCI',
        }
        self.request_count = 0
        self.rate_limit = Config.FRED_API_RATE_LIMIT
    
    def _rate_limit_check(self):
        """Simple rate limiting"""
        self.request_count += 1
        if self.request_count % 100 == 0:  # Be conservative
            logger.info("Rate limiting: sleeping 1 second")
            time.sleep(1)
    
    def get_indicator(self, series_id: str, start_date: datetime = None, end_date: datetime = None, retries: int = 3) -> Optional[pd.Series]:
        """Get single economic indicator with error handling and retries"""
        attempt = 0
        while attempt < retries:
            try:
                self._rate_limit_check()
                if start_date is None:
                    start_date = datetime.now() - timedelta(days=365*2)  # 2 years default
                if end_date is None:
                    end_date = datetime.now()
                data = self.fred.get_series(series_id, start=start_date, end=end_date)
                if data.empty:
                    logger.warning(f"No data returned for {series_id}")
                    return None
                logger.info(f"Successfully fetched {series_id}: {len(data)} data points")
                return data
            except Exception as e:
                logger.error(f"Error fetching {series_id} (attempt {attempt+1}): {e}", exc_info=True)
                time.sleep(2 ** attempt)
                attempt += 1
        logger.error(f"Failed to fetch {series_id} after {retries} attempts.")
        return None
    
    def get_all_indicators(self, start_date: datetime = None) -> pd.DataFrame:
        """Get all key economic indicators"""
        logger.info("Collecting all economic indicators")
        all_data = {}
        failed_indicators = []
        for name, series_id in self.key_indicators.items():
            logger.info(f"Fetching {name} ({series_id})")
            data = self.get_indicator(series_id, start_date)
            if data is not None:
                all_data[name] = data
            else:
                failed_indicators.append(name)
                logger.warning(f"Failed to fetch {name}")
        if failed_indicators:
            logger.warning(f"Failed to fetch {len(failed_indicators)} indicators: {failed_indicators}")
        if not all_data:
            logger.error("No economic data collected")
            print("[EconomicDataCollector] No economic data collected")
            return pd.DataFrame()
        economic_df = pd.DataFrame(all_data)
        economic_df = economic_df.fillna(method='ffill')
        economic_df = self._calculate_derived_indicators(economic_df)
        logger.info(f"Collected economic data: {economic_df.shape}")
        print(f"[EconomicDataCollector] Collected economic data: {economic_df.shape}")
        print(economic_df.head())
        return economic_df
    
    def _calculate_derived_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived economic indicators"""
        try:
            # Yield curve indicators
            if 'treasury_10y' in df.columns and 'treasury_2y' in df.columns:
                df['yield_curve_2_10'] = df['treasury_10y'] - df['treasury_2y']
            if 'treasury_10y' in df.columns and 'treasury_3m' in df.columns:
                df['yield_curve_3m_10y'] = df['treasury_10y'] - df['treasury_3m']
            # Inflation trends
            if 'cpi' in df.columns:
                df['cpi_yoy'] = df['cpi'].pct_change(periods=12) * 100  # Year-over-year
                df['cpi_mom'] = df['cpi'].pct_change() * 100  # Month-over-month
            # Real interest rates
            if 'fed_funds_rate' in df.columns and 'cpi_yoy' in df.columns:
                df['real_fed_funds'] = df['fed_funds_rate'] - df['cpi_yoy']
            # Labor market health
            if 'unemployment_rate' in df.columns:
                df['unemployment_change'] = df['unemployment_rate'].diff()
            # Economic momentum indicators
            for col in ['retail_sales', 'industrial_production', 'personal_income']:
                if col in df.columns:
                    df[f'{col}_3m_change'] = df[col].pct_change(periods=3) * 100
                    df[f'{col}_12m_change'] = df[col].pct_change(periods=12) * 100
            # Risk indicators
            if 'vix' in df.columns:
                df['vix_sma_20'] = df['vix'].rolling(window=20).mean()
                df['vix_spike'] = (df['vix'] > df['vix_sma_20'] * 1.5).astype(int)
            logger.info("Calculated derived economic indicators")
        except Exception as e:
            logger.error(f"Error calculating derived indicators: {e}")
        return df
    
    def get_economic_summary(self) -> Dict:
        """Get current economic summary"""
        economic_data = self.get_all_indicators()
        if economic_data.empty:
            return {'error': 'No economic data available'}
        latest = economic_data.iloc[-1]
        summary = {
            'data_date': economic_data.index[-1].strftime('%Y-%m-%d'),
            'gdp_growth': latest.get('GDP_growth', None),
            'unemployment_rate': latest.get('unemployment_rate', None),
            'inflation_rate': latest.get('cpi_yoy', None),
            'fed_funds_rate': latest.get('fed_funds_rate', None),
            'treasury_10y': latest.get('treasury_10y', None),
            'yield_curve_2_10': latest.get('yield_curve_2_10', None),
            'consumer_sentiment': latest.get('consumer_sentiment', None),
            'vix_level': latest.get('vix', None),
            'economic_indicators': {}
        }
        # Add trend analysis
        if len(economic_data) > 12:  # Need at least 12 months for trends
            summary['trends'] = self._analyze_trends(economic_data)
        # Economic health assessment
        summary['economic_health'] = self._assess_economic_health(latest)
        return summary
    
    def _analyze_trends(self, df: pd.DataFrame) -> Dict:
        """Analyze trends in economic data"""
        trends = {}
        try:
            recent = df.iloc[-3:]  # Last 3 months
            for col in ['unemployment_rate', 'cpi_yoy', 'fed_funds_rate', 'consumer_sentiment']:
                if col in df.columns:
                    values = recent[col].dropna()
                    if len(values) >= 2:
                        trend = 'rising' if values.iloc[-1] > values.iloc[0] else 'falling'
                        change = values.iloc[-1] - values.iloc[0]
                        trends[col] = {'trend': trend, 'change': change}
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
        return trends
    
    def _assess_economic_health(self, latest_data: pd.Series) -> str:
        """Simple economic health assessment"""
        try:
            unemployment = latest_data.get('unemployment_rate', 5)
            inflation = latest_data.get('cpi_yoy', 2)
            yield_curve = latest_data.get('yield_curve_2_10', 1)
            risk_factors = 0
            if unemployment > 6:
                risk_factors += 1
            if inflation > 4 or inflation < 0:
                risk_factors += 1
            if yield_curve < 0:  # Inverted yield curve
                risk_factors += 1
            if risk_factors == 0:
                return "Healthy"
            elif risk_factors == 1:
                return "Moderate Risk"
            else:
                return "High Risk"
        except Exception as e:
            logger.error(f"Error assessing economic health: {e}")
            return "Unknown"
    
    def save_economic_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """Save economic data to file"""
        if filename is None:
            filename = f"economic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = f"{Config.RAW_DATA_DIR}/{filename}"
        df.to_csv(filepath)
        logger.info(f"Saved economic data to {filepath}: {df.shape}")
        return filepath
