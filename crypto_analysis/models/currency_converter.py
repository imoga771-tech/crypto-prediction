import pandas as pd
from .api_client import CryptoAPIClient

class CurrencyConverter:
    """
    فئة لتحويل العملات الرقمية والتقليدية
    """
    def __init__(self, crypto_data=None):
        """
        تهيئة محول العملات
        
        المعلمات:
            crypto_data (dict): بيانات العملات الرقمية
        """
        self.crypto_data = crypto_data
        self.api_client = CryptoAPIClient()
        self.exchange_rates = self._get_exchange_rates()
    
    def _get_exchange_rates(self):
        """
        الحصول على أسعار صرف العملات
        
        العوائد:
            dict: أسعار صرف العملات
        """
        # الحصول على أسعار الصرف من API
        exchange_data = self.api_client.get_exchange_rates()
        
        # استخراج أسعار الصرف
        rates = exchange_data.get('rates', {})
        
        # التأكد من وجود العملات الأساسية
        if 'USD' not in rates:
            rates['USD'] = 1.0
        if 'SAR' not in rates:
            rates['SAR'] = 3.75
        if 'YER' not in rates:
            rates['YER'] = 250.0
        
        return rates
    
    def refresh_exchange_rates(self):
        """تحديث أسعار صرف العملات"""
        self.exchange_rates = self._get_exchange_rates()
    
    def refresh_crypto_prices(self):
        """تحديث أسعار العملات الرقمية"""
        if self.crypto_data is not None:
            # الحصول على أسعار العملات الرقمية من API
            api_prices = self.api_client.get_crypto_prices()
            
            # تحديث أسعار العملات الرقمية
            for symbol, data in api_prices.items():
                if symbol in self.crypto_data:
                    self.crypto_data[symbol]['current_price'] = data['current_price']
                    self.crypto_data[symbol]['price_change'] = data['price_change_percentage_24h']
    
    def convert(self, amount, from_currency, to_currency):
        """
        تحويل العملات
        
        المعلمات:
            amount (float): المبلغ
            from_currency (str): العملة المصدر
            to_currency (str): العملة الهدف
        
        العوائد:
            float: المبلغ المحول
        """
        # تحديث أسعار الصرف
        self.refresh_exchange_rates()
        
        # تحديث أسعار العملات الرقمية
        self.refresh_crypto_prices()
        
        # التحقق من صحة المدخلات
        if amount <= 0:
            return 0
        
        # تحويل من عملة رقمية إلى عملة تقليدية
        if from_currency in self.crypto_data and to_currency in self.exchange_rates:
            usd_value = amount * self.crypto_data[from_currency]['current_price']
            return usd_value * self.exchange_rates[to_currency]
        
        # تحويل من عملة تقليدية إلى عملة رقمية
        elif from_currency in self.exchange_rates and to_currency in self.crypto_data:
            usd_value = amount / self.exchange_rates[from_currency]
            return usd_value / self.crypto_data[to_currency]['current_price']
        
        # تحويل بين العملات التقليدية
        elif from_currency in self.exchange_rates and to_currency in self.exchange_rates:
            usd_value = amount / self.exchange_rates[from_currency]
            return usd_value * self.exchange_rates[to_currency]
        
        # تحويل بين العملات الرقمية
        elif from_currency in self.crypto_data and to_currency in self.crypto_data:
            usd_value = amount * self.crypto_data[from_currency]['current_price']
            return usd_value / self.crypto_data[to_currency]['current_price']
        
        return None
    
    def get_conversion_rate(self, from_currency, to_currency):
        """
        الحصول على سعر التحويل
        
        المعلمات:
            from_currency (str): العملة المصدر
            to_currency (str): العملة الهدف
        
        العوائد:
            float: سعر التحويل
        """
        return self.convert(1, from_currency, to_currency)
    
    def get_all_fiat_currencies(self):
        """
        الحصول على جميع العملات التقليدية
        
        العوائد:
            dict: العملات التقليدية
        """
        # تحديث أسعار الصرف
        self.refresh_exchange_rates()
        
        # إنشاء قاموس للعملات التقليدية
        fiat_currencies = {}
        
        # إضافة العملات الأساسية
        fiat_currencies['USD'] = 'دولار أمريكي'
        fiat_currencies['SAR'] = 'ريال سعودي'
        fiat_currencies['YER'] = 'ريال يمني'
        
        # إضافة العملات الأخرى
        for code in self.exchange_rates.keys():
            if code not in fiat_currencies and code not in ['USD', 'SAR', 'YER']:
                fiat_currencies[code] = code
        
        return fiat_currencies
    
    def get_popular_conversions(self, base_currency):
        """
        الحصول على التحويلات الشائعة
        
        المعلمات:
            base_currency (str): العملة الأساسية
        
        العوائد:
            list: قائمة بالتحويلات الشائعة
        """
        popular_conversions = []
        
        # تحديد العملات الشائعة
        popular_cryptos = ['BTC', 'ETH', 'BNB', 'TON']
        popular_fiats = ['USD', 'SAR', 'YER', 'EUR', 'GBP']
        
        # تحديد قائمة العملات المستهدفة
        if base_currency in self.crypto_data:
            target_currencies = popular_fiats
        else:
            target_currencies = popular_cryptos
        
        # إنشاء قائمة التحويلات
        for target in target_currencies:
            if target != base_currency:
                rate = self.get_conversion_rate(base_currency, target)
                if rate is not None:
                    popular_conversions.append({
                        'from_currency': base_currency,
                        'to_currency': target,
                        'rate': rate
                    })
        
        return popular_conversions