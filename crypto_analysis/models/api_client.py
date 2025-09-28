import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import json

class CryptoAPIClient:
    """
    فئة للتعامل مع واجهات برمجة التطبيقات الخارجية للعملات الرقمية
    """
    def __init__(self, cache_dir=None):
        """
        تهيئة العميل
        
        المعلمات:
            cache_dir (str): مسار مجلد التخزين المؤقت
        """
        self.coingecko_api_url = "https://api.coingecko.com/api/v3"
        self.exchange_rate_api_url = "https://open.er-api.com/v6/latest/USD"
        
        # إعداد مجلد التخزين المؤقت
        if cache_dir is None:
            self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')
        else:
            self.cache_dir = cache_dir
        
        # إنشاء مجلد التخزين المؤقت إذا لم يكن موجودًا
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # تعيين وقت انتهاء صلاحية التخزين المؤقت (بالثواني)
        self.cache_expiry = {
            'exchange_rates': 3600,  # ساعة واحدة
            'crypto_prices': 300,    # 5 دقائق
            'crypto_details': 86400  # يوم واحد
        }
        
        # قاموس لتحويل رموز العملات بين CoinGecko والتطبيق
        self.symbol_mapping = {
            # 'BTC': 'bitcoin',
            # 'ETH': 'ethereum',
            # 'BNB': 'binancecoin',
            # 'DOGE': 'dogecoin',
            # 'SOL': 'solana',
            # 'XRP': 'ripple',
            # 'ADA': 'cardano',
            # 'AVAX': 'avalanche-2',
            # 'LINK': 'chainlink',
            # 'LTC': 'litecoin',
            # 'BCH': 'bitcoin-cash',
            # 'TRX': 'tron',
            # 'XLM': 'stellar',
            # 'USDT': 'tether',
            # 'USDC': 'usd-coin',
            # 'HBAR': 'hedera-hashgraph'
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'BNB': 'binancecoin',
            'DOGE': 'dogecoin',
            'SOL': 'solana',
            'XRP': 'ripple',
            'ADA': 'cardano',
            'AVAX': 'avalanche-2',
            'LINK': 'chainlink',
            'LTC': 'litecoin',
            'BCH': 'bitcoin-cash',
            'TRX': 'tron',
            'DOT': 'polkadot',
            'LEO': 'leo-token',
            'TON': 'the-open-network',
            'SHIB': 'shiba-inu'
        }
        
        # القاموس العكسي
        self.reverse_symbol_mapping = {v: k for k, v in self.symbol_mapping.items()}

        # جلسة HTTP مع إعادة المحاولة وتهيئة مهلة افتراضية
        self.timeout = float(os.getenv("API_HTTP_TIMEOUT", "10"))
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
    
    def _get_cache_path(self, cache_type, key):
        """
        الحصول على مسار ملف التخزين المؤقت
        
        المعلمات:
            cache_type (str): نوع التخزين المؤقت
            key (str): مفتاح التخزين المؤقت
        
        العوائد:
            str: مسار ملف التخزين المؤقت
        """
        return os.path.join(self.cache_dir, f"{cache_type}_{key}.json")
    
    def _is_cache_valid(self, cache_path, cache_type):
        """
        التحقق مما إذا كان التخزين المؤقت صالحًا
        
        المعلمات:
            cache_path (str): مسار ملف التخزين المؤقت
            cache_type (str): نوع التخزين المؤقت
        
        العوائد:
            bool: True إذا كان التخزين المؤقت صالحًا، False خلاف ذلك
        """
        if not os.path.exists(cache_path):
            return False
        
        # التحقق من وقت التعديل
        file_time = os.path.getmtime(cache_path)
        current_time = time.time()
        
        return (current_time - file_time) < self.cache_expiry.get(cache_type, 3600)
    
    def _read_cache(self, cache_path):
        """
        قراءة البيانات من التخزين المؤقت
        
        المعلمات:
            cache_path (str): مسار ملف التخزين المؤقت
        
        العوائد:
            dict: البيانات المخزنة مؤقتًا
        """
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    
    def _write_cache(self, cache_path, data):
        """
        كتابة البيانات إلى التخزين المؤقت
        
        المعلمات:
            cache_path (str): مسار ملف التخزين المؤقت
            data (dict): البيانات للتخزين المؤقت
        """
        with open(cache_path, 'w') as f:
            json.dump(data, f)

    def _get_json(self, url, params=None):
        """
        طلب HTTP GET مع إعادة المحاولة وإرجاع JSON مع مهلة محددة
        """
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    def get_exchange_rates(self):
        """
        الحصول على أسعار صرف العملات
        
        العوائد:
            dict: أسعار صرف العملات
        """
        cache_path = self._get_cache_path('exchange_rates', 'latest')
        
        # التحقق من التخزين المؤقت
        if self._is_cache_valid(cache_path, 'exchange_rates'):
            cached_data = self._read_cache(cache_path)
            if cached_data:
                return cached_data
        
        try:
            # الحصول على أسعار الصرف من API
            data = self._get_json(self.exchange_rate_api_url)

            # تخزين البيانات مؤقتًا
            self._write_cache(cache_path, data)
            
            return data
        except requests.RequestException as e:
            print(f"خطأ في الحصول على أسعار الصرف: {e}")
            
            # محاولة استخدام التخزين المؤقت القديم إذا كان متاحًا
            cached_data = self._read_cache(cache_path)
            if cached_data:
                return cached_data
            
            # إرجاع أسعار صرف افتراضية إذا فشل كل شيء
            return {
                'rates': {
                    'USD': 1.0,
                    'SAR': 3.75,
                    'YER': 250.0
                },
                'time_last_updated': int(time.time())
            }
    
    def get_crypto_prices(self, symbols=None):
        """
        الحصول على أسعار العملات الرقمية
        
        المعلمات:
            symbols (list): قائمة برموز العملات
        
        العوائد:
            dict: أسعار العملات الرقمية
        """
        if symbols is None:
            symbols = list(self.symbol_mapping.keys())
        
        # تحويل رموز العملات إلى تنسيق CoinGecko
        coingecko_ids = [self.symbol_mapping.get(symbol, symbol.lower()) for symbol in symbols]
        coingecko_ids_str = ','.join(coingecko_ids)
        
        cache_path = self._get_cache_path('crypto_prices', 'latest')
        
        # التحقق من التخزين المؤقت
        if self._is_cache_valid(cache_path, 'crypto_prices'):
            cached_data = self._read_cache(cache_path)
            if cached_data:
                return cached_data
        
        try:
            # الحصول على أسعار العملات من API
            url = f"{self.coingecko_api_url}/coins/markets"
            params = {
                'vs_currency': 'usd',
                'ids': coingecko_ids_str,
                'order': 'market_cap_desc',
                'per_page': 100,
                'page': 1,
                'sparkline': False,
                'price_change_percentage': '24h'
            }
            
            data = self._get_json(url, params=params)
            
            # تحويل البيانات إلى تنسيق التطبيق
            formatted_data = {}
            for coin in data:
                symbol = self.reverse_symbol_mapping.get(coin['id'], coin['symbol'].upper())
                formatted_data[symbol] = {
                    'name': coin['name'],
                    'symbol': symbol,
                    'current_price': coin['current_price'],
                    'price_change_percentage_24h': coin['price_change_percentage_24h'],
                    'market_cap': coin['market_cap'],
                    'total_volume': coin['total_volume'],
                    'image': coin['image'],
                    'last_updated': coin['last_updated']
                }
            
            # تخزين البيانات مؤقتًا
            self._write_cache(cache_path, formatted_data)
            
            return formatted_data
        except requests.RequestException as e:
            print(f"خطأ في الحصول على أسعار العملات الرقمية: {e}")
            
            # محاولة استخدام التخزين المؤقت القديم إذا كان متاحًا
            cached_data = self._read_cache(cache_path)
            if cached_data:
                return cached_data
            
            # إرجاع بيانات فارغة إذا فشل كل شيء
            return {}
    
    def get_crypto_historical_data(self, symbol, days=30):
        """
        الحصول على البيانات التاريخية للعملة الرقمية
        
        المعلمات:
            symbol (str): رمز العملة
            days (int): عدد الأيام
        
        العوائد:
            DataFrame: البيانات التاريخية
        """
        # تحويل رمز العملة إلى تنسيق CoinGecko
        coingecko_id = self.symbol_mapping.get(symbol, symbol.lower())
        
        cache_path = self._get_cache_path('crypto_historical', f"{symbol}_{days}")
        
        # التحقق من التخزين المؤقت
        if self._is_cache_valid(cache_path, 'crypto_prices'):
            cached_data = self._read_cache(cache_path)
            if cached_data:
                return pd.DataFrame(cached_data)
        
        try:
            # الحصول على البيانات التاريخية من API
            url = f"{self.coingecko_api_url}/coins/{coingecko_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily'
            }
            
            data = self._get_json(url, params=params)
            
            # تحويل البيانات إلى DataFrame
            prices = data['prices']
            volumes = data.get('total_volumes', [[0, 0]] * len(prices))
            
            df_data = []
            for i, (price_data, volume_data) in enumerate(zip(prices, volumes)):
                timestamp, price = price_data
                _, volume = volume_data
                
                date = datetime.fromtimestamp(timestamp / 1000)
                
                df_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'close': price,
                    'volume': volume
                })
            
            # تخزين البيانات مؤقتًا
            self._write_cache(cache_path, df_data)
            
            return pd.DataFrame(df_data)
        except requests.RequestException as e:
            print(f"خطأ في الحصول على البيانات التاريخية: {e}")
            
            # محاولة استخدام التخزين المؤقت القديم إذا كان متاحًا
            cached_data = self._read_cache(cache_path)
            if cached_data:
                return pd.DataFrame(cached_data)
            
            # إرجاع DataFrame فارغ إذا فشل كل شيء
            return pd.DataFrame(columns=['date', 'close', 'volume'])
    
    def get_crypto_details(self, symbol):
        """
        الحصول على تفاصيل العملة الرقمية
        
        المعلمات:
            symbol (str): رمز العملة
        
        العوائد:
            dict: تفاصيل العملة
        """
        # تحويل رمز العملة إلى تنسيق CoinGecko
        coingecko_id = self.symbol_mapping.get(symbol, symbol.lower())
        
        cache_path = self._get_cache_path('crypto_details', symbol)
        
        # التحقق من التخزين المؤقت
        if self._is_cache_valid(cache_path, 'crypto_details'):
            cached_data = self._read_cache(cache_path)
            if cached_data:
                return cached_data
        
        try:
            # الحصول على تفاصيل العملة من API
            url = f"{self.coingecko_api_url}/coins/{coingecko_id}"
            params = {
                'localization': 'false',
                'tickers': 'false',
                'market_data': 'true',
                'community_data': 'false',
                'developer_data': 'false'
            }
            
            data = self._get_json(url, params=params)
            
            # تحويل البيانات إلى تنسيق التطبيق
            formatted_data = {
                'name': data['name'],
                'symbol': symbol,
                'description': data['description'].get('ar', data['description'].get('en', '')),
                'image': data['image']['large'],
                'current_price': data['market_data']['current_price']['usd'],
                'market_cap': data['market_data']['market_cap']['usd'],
                'total_volume': data['market_data']['total_volume']['usd'],
                'high_24h': data['market_data']['high_24h']['usd'],
                'low_24h': data['market_data']['low_24h']['usd'],
                'price_change_percentage_24h': data['market_data']['price_change_percentage_24h'],
                'price_change_percentage_7d': data['market_data']['price_change_percentage_7d'],
                'price_change_percentage_30d': data['market_data']['price_change_percentage_30d'],
                'last_updated': data['last_updated']
            }
            
            # تخزين البيانات مؤقتًا
            self._write_cache(cache_path, formatted_data)
            
            return formatted_data
        except requests.RequestException as e:
            print(f"خطأ في الحصول على تفاصيل العملة: {e}")
            
            # محاولة استخدام التخزين المؤقت القديم إذا كان متاحًا
            cached_data = self._read_cache(cache_path)
            if cached_data:
                return cached_data
            
            # إرجاع بيانات فارغة إذا فشل كل شيء
            return {
                'name': symbol,
                'symbol': symbol,
                'description': 'لا تتوفر معلومات',
                'image': '',
                'current_price': 0,
                'market_cap': 0,
                'total_volume': 0,
                'high_24h': 0,
                'low_24h': 0,
                'price_change_percentage_24h': 0,
                'price_change_percentage_7d': 0,
                'price_change_percentage_30d': 0,
                'last_updated': datetime.now().isoformat()
            }
    
    def merge_api_data_with_csv(self, symbol, csv_data):
        """
        دمج بيانات API مع بيانات CSV
        
        المعلمات:
            symbol (str): رمز العملة
            csv_data (DataFrame): بيانات CSV
        
        العوائد:
            DataFrame: البيانات المدمجة
        """
        # الحصول على البيانات التاريخية من API
        api_data = self.get_crypto_historical_data(symbol, days=30)
        
        if api_data.empty:
            return csv_data
        
        # تحويل عمود التاريخ إلى datetime
        api_data['date'] = pd.to_datetime(api_data['date'])
        
        # التأكد من أن بيانات CSV تحتوي على عمود التاريخ بتنسيق datetime
        csv_data['date'] = pd.to_datetime(csv_data['date'])
        
        # تحديد آخر تاريخ في بيانات CSV
        last_csv_date = csv_data['date'].max()
        
        # تصفية بيانات API للحصول على البيانات الجديدة فقط
        new_api_data = api_data[api_data['date'] > last_csv_date]
        
        if new_api_data.empty:
            return csv_data
        
        # إضافة الأعمدة المفقودة إلى بيانات API
        for col in csv_data.columns:
            if col not in new_api_data.columns:
                if col == 'open':
                    new_api_data['open'] = new_api_data['close'].shift(1)
                elif col == 'high':
                    new_api_data['high'] = new_api_data['close'] * 1.01  # تقريب
                elif col == 'low':
                    new_api_data['low'] = new_api_data['close'] * 0.99  # تقريب
                else:
                    new_api_data[col] = 0
        
        # دمج البيانات
        merged_data = pd.concat([csv_data, new_api_data], ignore_index=True)
        
        # ملء القيم المفقودة
        merged_data = merged_data.fillna(method='ffill')
        
        return merged_data