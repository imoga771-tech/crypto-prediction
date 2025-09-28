import os
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd

from .models.api_client import CryptoAPIClient
from .models.currency_converter import CurrencyConverter

logger = logging.getLogger(__name__)

# Singletons (encapsulated to avoid scattering globals across app)
_api_client: Optional[CryptoAPIClient] = None
_crypto_data: Optional[Dict[str, Dict[str, Any]]] = None
_crypto_loaded_at: Optional[float] = None
_currency_converter: Optional[CurrencyConverter] = None


def get_api_client() -> CryptoAPIClient:
    global _api_client
    if _api_client is None:
        _api_client = CryptoAPIClient()
    return _api_client


def _crypto_files_path() -> str:
    # coin files csv directory at project root
    # crypto_analysis/services.py -> up one (crypto_analysis), up one (root) -> coin files csv
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'coin files csv')


def load_crypto_data() -> Dict[str, Dict[str, Any]]:
    """
    Load crypto data from CSVs and enrich with API where possible.
    Mirrors the previous logic in app.py but encapsulated and reusable.
    """
    logger.info("بدء تحميل بيانات العملات الرقمية (services.load_crypto_data)")
    start_time = time.time()

    api_client = get_api_client()

    crypto_data: Dict[str, Dict[str, Any]] = {}
    crypto_info = {
        'BTC': {'name': 'Bitcoin', 'color': '#F7931A', 'description': 'أول وأشهر عملة رقمية في العالم'},
        'ETH': {'name': 'Ethereum', 'color': '#627EEA', 'description': 'منصة للعقود الذكية والتطبيقات اللامركزية'},
        'BNB': {'name': 'Binance Coin', 'color': '#F3BA2F', 'description': 'العملة الأساسية لمنصة بينانس'},
        'DOGE': {'name': 'Dogecoin', 'color': '#C2A633', 'description': 'عملة رقمية بدأت كمزحة وأصبحت شائعة'},
        'SOL': {'name': 'Solana', 'color': '#00FFA3', 'description': 'منصة عالية الأداء للتطبيقات اللامركزية'},
        'XRP': {'name': 'Ripple', 'color': '#23292F', 'description': 'نظام دفع رقمي وشبكة صرف عملات'},
        'ADA': {'name': 'Cardano', 'color': '#0033AD', 'description': 'منصة للعقود الذكية مع نهج علمي'},
        'AVAX': {'name': 'Avalanche', 'color': '#E84142', 'description': 'منصة للتطبيقات اللامركزية والأصول المخصصة'},
        'LINK': {'name': 'Chainlink', 'color': '#2A5ADA', 'description': 'شبكة لامركزية للعقود الذكية'},
        'LTC': {'name': 'Litecoin', 'color': '#BFBBBB', 'description': 'عملة رقمية مصممة لتكون أسرع من البيتكوين'},
        'BCH': {'name': 'Bitcoin Cash', 'color': '#8DC351', 'description': 'نسخة معدلة من البيتكوين لمعاملات أسرع'},
        'TRX': {'name': 'TRON', 'color': '#EF0027', 'description': 'منصة لامركزية للترفيه الرقمي'},
        'DOT': {'name': 'Polkadot', 'color': '#E6007A', 'description': 'شبكة متعددة السلاسل لربط البلوكشين'},
        'LEO': {'name': 'LEO Token', 'color': '#FCB500', 'description': 'رمز منصة iFinex'},
        'TON': {'name': 'The Open Network', 'color': '#1DA1F2', 'description': 'شبكة لامركزية لتطبيقات WEB3'},
        'SHIB': {'name': 'Shiba Inu', 'color': '#F5A623', 'description': 'عملة مزحة أصبحت شائعة'}

        # 'BTC': {'name': 'Bitcoin', 'color': '#F7931A', 'description': 'أول وأشهر عملة رقمية في العالم'},
        # 'ETH': {'name': 'Ethereum', 'color': '#627EEA', 'description': 'منصة للعقود الذكية والتطبيقات اللامركزية'},
        # 'BNB': {'name': 'Binance Coin', 'color': '#F3BA2F', 'description': 'العملة الأساسية لمنصة بينانس'},
        # 'DOGE': {'name': 'Dogecoin', 'color': '#C2A633', 'description': 'عملة رقمية بدأت كمزحة وأصبحت شائعة'},
        # 'SOL': {'name': 'Solana', 'color': '#00FFA3', 'description': 'منصة عالية الأداء للتطبيقات اللامركزية'},
        # 'XRP': {'name': 'Ripple', 'color': '#23292F', 'description': 'نظام دفع رقمي وشبكة صرف عملات'},
        # 'ADA': {'name': 'Cardano', 'color': '#0033AD', 'description': 'منصة للعقود الذكية مع نهج علمي'},
        # 'AVAX': {'name': 'Avalanche', 'color': '#E84142', 'description': 'منصة للتطبيقات اللامركزية والأصول المخصصة'},
        # 'LINK': {'name': 'Chainlink', 'color': '#2A5ADA', 'description': 'شبكة لامركزية للعقود الذكية'},
        # 'LTC': {'name': 'Litecoin', 'color': '#BFBBBB', 'description': 'عملة رقمية مصممة لتكون أسرع من البيتكوين'},
        # 'BCH': {'name': 'Bitcoin Cash', 'color': '#8DC351', 'description': 'نسخة معدلة من البيتكوين لمعاملات أسرع'},
        # 'TRX': {'name': 'TRON', 'color': '#EF0027', 'description': 'منصة لامركزية للترفيه الرقمي'},
        # 'XLM': {'name': 'Stellar', 'color': '#14B6E7', 'description': 'منصة للمدفوعات الرقمية والتحويلات'},
        # 'USDT': {'name': 'Tether', 'color': '#26A17B', 'description': 'عملة مستقرة مرتبطة بالدولار الأمريكي'},
        # 'USDC': {'name': 'USD Coin', 'color': '#2775CA', 'description': 'عملة مستقرة مرتبطة بالدولار الأمريكي'},
        # 'HBAR': {'name': 'Hedera', 'color': '#222222', 'description': 'شبكة لامركزية للتطبيقات والمدفوعات'}
    }

    # Fetch API prices once (with caching inside client)
    try:
        api_prices = api_client.get_crypto_prices()
        logger.info(f"تم الحصول على أسعار {len(api_prices)} عملة من API")
    except Exception as e:
        logger.error(f"خطأ في الحصول على أسعار العملات من API: {e}")
        api_prices = {}

    csv_dir = _crypto_files_path()
    if not os.path.isdir(csv_dir):
        logger.warning(f"مسار ملفات CSV غير موجود: {csv_dir}")

    # Validate required columns set
    required_cols = {'date', 'close'}
    optional_ohlcv = {'open', 'high', 'low', 'volume'}

    for filename in os.listdir(csv_dir) if os.path.isdir(csv_dir) else []:
        if not filename.endswith('.csv'):
            continue
        symbol = filename.split('.')[0]
        file_path = os.path.join(csv_dir, filename)

        try:
            df = pd.read_csv(file_path)

            # Basic required columns
            missing = required_cols - set(df.columns)
            if missing:
                logger.warning(f"ملف {filename} يفتقد أعمدة مطلوبة: {missing} - سيتم تجاهله")
                continue

            # Normalize date
            df['date'] = pd.to_datetime(df['date'])

            # Ensure optional columns exist to keep downstream indicators happy
            for col in optional_ohlcv:
                if col not in df.columns:
                    if col == 'open':
                        df['open'] = df['close'].shift(1).fillna(df['close'])
                    elif col == 'high':
                        df['high'] = (df['close'] * 1.01).astype(float)
                    elif col == 'low':
                        df['low'] = (df['close'] * 0.99).astype(float)
                    elif col == 'volume':
                        df['volume'] = 0.0

            # Keep a pure CSV copy for charts (no API merge)
            df_csv_only = df.copy().sort_values('date').reset_index(drop=True)

            # Merge recent API historical data (30d) if available
            try:
                if symbol in api_prices:
                    df = api_client.merge_api_data_with_csv(symbol, df)
            except Exception as merge_err:
                logger.warning(f"تعذر دمج بيانات API مع CSV للعملة {symbol}: {merge_err}")

            info = crypto_info.get(symbol, {'name': symbol, 'color': '#808080', 'description': 'عملة رقمية'})

            latest_data = df.iloc[-1]
            current_price = float(latest_data['close'])
            price_change = 0.0

            if symbol in api_prices:
                current_price = api_prices[symbol].get('current_price', current_price)
                price_change = api_prices[symbol].get('price_change_percentage_24h', 0.0)
            else:
                if len(df) > 1:
                    prev_close = df.iloc[-2]['close']
                    if prev_close != 0:
                        price_change = (latest_data['close'] - prev_close) / prev_close * 100

            # Try to fetch details (sanitized inside api_client)
            try:
                crypto_details = api_client.get_crypto_details(symbol)
                if crypto_details and crypto_details.get('description'):
                    info['description'] = crypto_details['description']
            except Exception as e:
                logger.warning(f"لم يتم الحصول على تفاصيل إضافية للعملة {symbol}: {e}")

            crypto_data[symbol] = {
                'name': info['name'],
                'symbol': symbol,
                'color': info['color'],
                'description': info['description'],
                'current_price': current_price,
                'price_change': price_change,
                'last_updated': datetime.now() if symbol in api_prices else latest_data['date'],
                'data': df.sort_values('date').reset_index(drop=True),
                'data_csv': df_csv_only,
                'market_cap': api_prices.get(symbol, {}).get('market_cap', 0),
                'total_volume': api_prices.get(symbol, {}).get('total_volume', 0),
                'image': api_prices.get(symbol, {}).get('image', '')
            }

            logger.info(f"تم تحميل بيانات العملة {symbol} بنجاح")
        except Exception as e:
            logger.error(f"خطأ في قراءة/تحضير ملف {filename}: {e}")

    end_time = time.time()
    logger.info(f"تم تحميل بيانات {len(crypto_data)} عملة في {end_time - start_time:.2f} ثانية")

    return crypto_data


def get_crypto_data(force_reload: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Lazy singleton access to crypto_data. Loads once and caches.
    Use force_reload=True to refresh from disk/API.
    """
    global _crypto_data, _crypto_loaded_at, _currency_converter
    if force_reload or _crypto_data is None:
        _crypto_data = load_crypto_data()
        _crypto_loaded_at = time.time()
        # Invalidate converter to rebuild with fresh data
        _currency_converter = None
    return _crypto_data


def get_currency_converter() -> CurrencyConverter:
    """
    Returns a CurrencyConverter built on current crypto_data, refreshing exchange
    rates opportunistically.
    """
    global _currency_converter
    if _currency_converter is None:
        _currency_converter = CurrencyConverter(get_crypto_data())
        try:
            _currency_converter.refresh_exchange_rates()
        except Exception as e:
            logger.warning(f"تعذر تحديث أسعار الصرف عند الإنشاء: {e}")
    return _currency_converter


def refresh_exchange_rates_safe() -> None:
    try:
        get_currency_converter().refresh_exchange_rates()
    except Exception as e:
        logger.error(f"خطأ أثناء تحديث أسعار الصرف: {e}")


def available_symbols() -> Dict[str, str]:
    data = get_crypto_data()
    return {symbol: meta['name'] for symbol, meta in data.items()}


def clamp_days(value: int, min_days: int = 1, max_days: int = 120) -> int:
    try:
        v = int(value)
    except Exception:
        v = min_days
    return max(min_days, min(max_days, v))


def valid_model_type(model_type: str) -> str:
    allowed = {'linear', 'rf', 'svr', 'lstm', 'ensemble'}
    return model_type if model_type in allowed else 'ensemble'


def valid_currency(code: str) -> str:
    conv = get_currency_converter()
    rates = conv.get_all_fiat_currencies()
    return code if code in rates else 'USD'


def update_prices_in_place() -> Dict[str, Any]:
    """
    Fetch latest market data and update in-memory crypto_data safely.
    Returns a summary dict for API responses.
    """
    data = get_crypto_data()
    api_client = get_api_client()
    updated = 0
    try:
        api_prices = api_client.get_crypto_prices()
        for symbol, p in api_prices.items():
            if symbol in data:
                data[symbol]['current_price'] = p.get('current_price', data[symbol]['current_price'])
                data[symbol]['price_change'] = p.get('price_change_percentage_24h', data[symbol]['price_change'])
                data[symbol]['market_cap'] = p.get('market_cap', data[symbol]['market_cap'])
                data[symbol]['total_volume'] = p.get('total_volume', data[symbol]['total_volume'])
                data[symbol]['image'] = p.get('image', data[symbol]['image'])
                data[symbol]['last_updated'] = datetime.now()
                updated += 1
    except Exception as e:
        logger.error(f"خطأ في تحديث أسعار العملات: {e}")
        raise
    return {'updated': updated, 'timestamp': datetime.now().isoformat()}