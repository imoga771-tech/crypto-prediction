import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class TechnicalIndicators:
    """
    فئة لحساب وعرض المؤشرات الفنية للعملات الرقمية
    """
    def __init__(self, data):
        """
        تهيئة المؤشرات الفنية
        
        المعلمات:
            data (DataFrame): بيانات العملة
        """
        self.data = data.copy()
    
    def add_all_indicators(self):
        """
        إضافة جميع المؤشرات الفنية إلى البيانات
        
        العوائد:
            DataFrame: البيانات مع المؤشرات الفنية
        """
        df = self.data.copy()
        
        # إضافة المتوسطات المتحركة
        df = self.add_moving_averages(df)
        
        # إضافة مؤشر القوة النسبية
        df = self.add_rsi(df)
        
        # إضافة مؤشر تقارب وتباعد المتوسطات المتحركة
        df = self.add_macd(df)
        
        # إضافة نطاقات بولينجر
        df = self.add_bollinger_bands(df)
        
        # إضافة مؤشر تدفق الأموال
        df = self.add_money_flow_index(df)
        
        # إضافة مؤشر القناة السعرية
        df = self.add_price_channels(df)
        
        # إضافة مؤشر الزخم
        df = self.add_momentum(df)
        
        # إضافة مؤشر ستوكاستيك
        df = self.add_stochastic_oscillator(df)
        
        # حذف الصفوف التي تحتوي على قيم NaN
        df = df.dropna().reset_index(drop=True)
        
        return df
    
    def add_moving_averages(self, df):
        """
        إضافة المتوسطات المتحركة
        
        المعلمات:
            df (DataFrame): البيانات
        
        العوائد:
            DataFrame: البيانات مع المتوسطات المتحركة
        """
        # إضافة المتوسطات المتحركة البسيطة
        df['SMA7'] = df['close'].rolling(window=7).mean()
        df['SMA14'] = df['close'].rolling(window=14).mean()
        df['SMA30'] = df['close'].rolling(window=30).mean()
        df['SMA50'] = df['close'].rolling(window=50).mean()
        df['SMA200'] = df['close'].rolling(window=200).mean()
        
        # إضافة المتوسطات المتحركة الأسية
        df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        return df
    
    def add_rsi(self, df, period=14):
        """
        إضافة مؤشر القوة النسبية
        
        المعلمات:
            df (DataFrame): البيانات
            period (int): الفترة
        
        العوائد:
            DataFrame: البيانات مع مؤشر القوة النسبية
        """
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df
    
    def add_macd(self, df):
        """
        إضافة مؤشر تقارب وتباعد المتوسطات المتحركة
        
        المعلمات:
            df (DataFrame): البيانات
        
        العوائد:
            DataFrame: البيانات مع مؤشر MACD
        """
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        return df
    
    def add_bollinger_bands(self, df, period=20, std_dev=2):
        """
        إضافة نطاقات بولينجر
        
        المعلمات:
            df (DataFrame): البيانات
            period (int): الفترة
            std_dev (int): عدد الانحرافات المعيارية
        
        العوائد:
            DataFrame: البيانات مع نطاقات بولينجر
        """
        df['BB_middle'] = df['close'].rolling(window=period).mean()
        df['BB_std'] = df['close'].rolling(window=period).std()
        df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * std_dev)
        df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * std_dev)
        
        return df
    
    def add_money_flow_index(self, df, period=14):
        """
        إضافة مؤشر تدفق الأموال
        
        المعلمات:
            df (DataFrame): البيانات
            period (int): الفترة
        
        العوائد:
            DataFrame: البيانات مع مؤشر تدفق الأموال
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume'] if 'volume' in df.columns else typical_price
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).fillna(0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).fillna(0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi_ratio = positive_mf / negative_mf
        df['MFI'] = 100 - (100 / (1 + mfi_ratio))
        
        return df
    
    def add_price_channels(self, df, period=20):
        """
        إضافة مؤشر القناة السعرية
        
        المعلمات:
            df (DataFrame): البيانات
            period (int): الفترة
        
        العوائد:
            DataFrame: البيانات مع مؤشر القناة السعرية
        """
        df['PC_upper'] = df['high'].rolling(window=period).max()
        df['PC_lower'] = df['low'].rolling(window=period).min()
        df['PC_middle'] = (df['PC_upper'] + df['PC_lower']) / 2
        
        return df
    
    def add_momentum(self, df, period=14):
        """
        إضافة مؤشر الزخم
        
        المعلمات:
            df (DataFrame): البيانات
            period (int): الفترة
        
        العوائد:
            DataFrame: البيانات مع مؤشر الزخم
        """
        df['Momentum'] = df['close'] - df['close'].shift(period)
        
        return df
    
    def add_stochastic_oscillator(self, df, k_period=14, d_period=3):
        """
        إضافة مؤشر ستوكاستيك
        
        المعلمات:
            df (DataFrame): البيانات
            k_period (int): فترة K
            d_period (int): فترة D
        
        العوائد:
            DataFrame: البيانات مع مؤشر ستوكاستيك
        """
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        df['Stoch_K'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=d_period).mean()
        
        return df
    
    def create_technical_charts(self, symbol, name):
        """
        إنشاء رسوم بيانية للمؤشرات الفنية
        
        المعلمات:
            symbol (str): رمز العملة
            name (str): اسم العملة
        
        العوائد:
            dict: الرسوم البيانية
        """
        # إضافة المؤشرات الفنية
        df = self.add_all_indicators()
        
        # إنشاء الرسوم البيانية
        charts = {}
        
        # رسم بياني للسعر مع المتوسطات المتحركة
        charts['price_ma'] = self._create_price_ma_chart(df, symbol, name)
        
        # رسم بياني للسعر مع نطاقات بولينجر
        charts['bollinger'] = self._create_bollinger_chart(df, symbol, name)
        
        # رسم بياني لمؤشر القوة النسبية
        charts['rsi'] = self._create_rsi_chart(df, symbol, name)
        
        # رسم بياني لمؤشر MACD
        charts['macd'] = self._create_macd_chart(df, symbol, name)
        
        # رسم بياني لمؤشر ستوكاستيك
        charts['stochastic'] = self._create_stochastic_chart(df, symbol, name)
        
        # رسم بياني لحجم التداول
        if 'volume' in df.columns:
            charts['volume'] = self._create_volume_chart(df, symbol, name)
        
        return charts
    
    def _create_price_ma_chart(self, df, symbol, name):
        """
        إنشاء رسم بياني للسعر مع المتوسطات المتحركة
        
        المعلمات:
            df (DataFrame): البيانات
            symbol (str): رمز العملة
            name (str): اسم العملة
        
        العوائد:
            Figure: الرسم البياني
        """
        fig = go.Figure()
        
        # إضافة خط السعر
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['close'],
            mode='lines',
            name='السعر',
            line=dict(color='#2c3e50', width=2)
        ))
        
        # إضافة المتوسطات المتحركة
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['SMA7'],
            mode='lines',
            name='SMA 7',
            line=dict(color='#3498db', width=1.5)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['SMA30'],
            mode='lines',
            name='SMA 30',
            line=dict(color='#2ecc71', width=1.5)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['SMA200'],
            mode='lines',
            name='SMA 200',
            line=dict(color='#e74c3c', width=1.5)
        ))
        
        # تحديث تخطيط الرسم البياني
        fig.update_layout(
            title=f"سعر {name} ({symbol}) مع المتوسطات المتحركة",
            xaxis_title="التاريخ",
            yaxis_title="السعر (USD)",
            template='plotly_white',
            plot_bgcolor='rgba(255, 255, 255, 1)',
            paper_bgcolor='rgba(255, 255, 255, 1)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig

    def _create_volume_chart(self, df, symbol, name):
        """
        إنشاء رسم بياني لحجم التداول
        
        المعلمات:
            df (DataFrame): البيانات
            symbol (str): رمز العملة
            name (str): اسم العملة
        
        العوائد:
            Figure: الرسم البياني
        """
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df['date'],
            y=df['volume'],
            name='الحجم',
            marker_color='#3498db'
        ))
        fig.update_layout(
            title=f"حجم التداول لـ {name} ({symbol})",
            xaxis_title="التاريخ",
            yaxis_title="الحجم",
            template='plotly_white',
            plot_bgcolor='rgba(255, 255, 255, 1)',
            paper_bgcolor='rgba(255, 255, 255, 1)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        return fig
    
    def _create_bollinger_chart(self, df, symbol, name):
        """
        إنشاء رسم بياني للسعر مع نطاقات بولينجر
        
        المعلمات:
            df (DataFrame): البيانات
            symbol (str): رمز العملة
            name (str): اسم العملة
        
        العوائد:
            Figure: الرسم البياني
        """
        fig = go.Figure()
        
        # إضافة خط السعر
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['close'],
            mode='lines',
            name='السعر',
            line=dict(color='#2c3e50', width=2)
        ))
        
        # إضافة نطاقات بولينجر
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['BB_upper'],
            mode='lines',
            name='النطاق العلوي',
            line=dict(color='#e74c3c', width=1.5)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['BB_middle'],
            mode='lines',
            name='المتوسط',
            line=dict(color='#3498db', width=1.5)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['BB_lower'],
            mode='lines',
            name='النطاق السفلي',
            line=dict(color='#2ecc71', width=1.5),
            fill='tonexty',
            fillcolor='rgba(46, 204, 113, 0.1)'
        ))
        
        # تحديث تخطيط الرسم البياني
        fig.update_layout(
            title=f"نطاقات بولينجر لـ {name} ({symbol})",
            xaxis_title="التاريخ",
            yaxis_title="السعر (USD)",
            template='plotly_white',
            plot_bgcolor='rgba(255, 255, 255, 1)',
            paper_bgcolor='rgba(255, 255, 255, 1)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def _create_rsi_chart(self, df, symbol, name):
        """
        إنشاء رسم بياني لمؤشر القوة النسبية
        """
        # حراسة: لا ترسم إذا كانت البيانات غير كافية
        if df is None or df.empty or 'date' not in df.columns or 'RSI' not in df.columns or df['RSI'].dropna().empty:
            fig = go.Figure()
            fig.update_layout(
                title=f"مؤشر القوة النسبية (RSI) لـ {name} ({symbol}) - لا توجد بيانات كافية",
                xaxis_title="التاريخ",
                yaxis_title="RSI",
                template='plotly_white',
                plot_bgcolor='rgba(255, 255, 255, 1)',
                paper_bgcolor='rgba(255, 255, 255, 1)'
            )
            return fig

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['RSI'], mode='lines', name='RSI', line=dict(color='#3498db', width=2)
        ))

        # خطوط المستويات والAnnotations بشكل آمن
        x0 = df['date'].iloc[0]
        x1 = df['date'].iloc[-1]
        fig.add_shape(type="line", x0=x0, y0=70, x1=x1, y1=70, line=dict(color="#e74c3c", width=1, dash="dash"))
        fig.add_shape(type="line", x0=x0, y0=30, x1=x1, y1=30, line=dict(color="#2ecc71", width=1, dash="dash"))

        fig.update_layout(
            title=f"مؤشر القوة النسبية (RSI) لـ {name} ({symbol})",
            xaxis_title="التاريخ",
            yaxis_title="RSI",
            yaxis=dict(range=[0, 100]),
            template='plotly_white', plot_bgcolor='rgba(255,255,255,1)', paper_bgcolor='rgba(255,255,255,1)'
        )

        fig.add_annotation(x=x1, y=70, text="ذروة الشراء", showarrow=False, yshift=10)
        fig.add_annotation(x=x1, y=30, text="ذروة البيع", showarrow=False, yshift=-10)

        return fig
    
    def _create_macd_chart(self, df, symbol, name):
        """
        إنشاء رسم بياني لمؤشر MACD
        
        المعلمات:
            df (DataFrame): البيانات
            symbol (str): رمز العملة
            name (str): اسم العملة
        
        العوائد:
            Figure: الرسم البياني
        """
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.1, row_heights=[0.7, 0.3])
        
        # إضافة خط السعر
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['close'],
                mode='lines',
                name='السعر',
                line=dict(color='#2c3e50', width=2)
            ),
            row=1, col=1
        )
        
        # إضافة مؤشر MACD
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='#3498db', width=1.5)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['MACD_signal'],
                mode='lines',
                name='إشارة MACD',
                line=dict(color='#e74c3c', width=1.5)
            ),
            row=2, col=1
        )
        
        # إضافة المدرج التكراري
        colors = ['#2ecc71' if val >= 0 else '#e74c3c' for val in df['MACD_histogram']]
        
        fig.add_trace(
            go.Bar(
                x=df['date'],
                y=df['MACD_histogram'],
                name='المدرج التكراري',
                marker_color=colors
            ),
            row=2, col=1
        )
        
        # تحديث تخطيط الرسم البياني
        fig.update_layout(
            title=f"مؤشر MACD لـ {name} ({symbol})",
            template='plotly_white',
            plot_bgcolor='rgba(255, 255, 255, 1)',
            paper_bgcolor='rgba(255, 255, 255, 1)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_yaxes(title_text="السعر (USD)", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        fig.update_xaxes(title_text="التاريخ", row=2, col=1)
        
        return fig
    
    def _create_stochastic_chart(self, df, symbol, name):
        """
        إنشاء رسم بياني لمؤشر ستوكاستيك
        """
        # حراسة: لا ترسم إذا كانت البيانات غير كافية
        if (
            df is None or df.empty or 'date' not in df.columns or
            'Stoch_K' not in df.columns or 'Stoch_D' not in df.columns or
            df['Stoch_K'].dropna().empty or df['Stoch_D'].dropna().empty
        ):
            fig = go.Figure()
            fig.update_layout(
                title=f"مؤشر ستوكاستيك لـ {name} ({symbol}) - لا توجد بيانات كافية",
                xaxis_title="التاريخ",
                yaxis_title="ستوكاستيك",
                template='plotly_white',
                plot_bgcolor='rgba(255,255,255,1)', paper_bgcolor='rgba(255,255,255,1)'
            )
            return fig

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['date'], y=df['Stoch_K'], mode='lines', name='%K', line=dict(color='#3498db', width=2)))
        fig.add_trace(go.Scatter(x=df['date'], y=df['Stoch_D'], mode='lines', name='%D', line=dict(color='#e74c3c', width=2)))

        x0 = df['date'].iloc[0]
        x1 = df['date'].iloc[-1]
        fig.add_shape(type="line", x0=x0, y0=80, x1=x1, y1=80, line=dict(color="#e74c3c", width=1, dash="dash"))
        fig.add_shape(type="line", x0=x0, y0=20, x1=x1, y1=20, line=dict(color="#2ecc71", width=1, dash="dash"))

        fig.update_layout(
            title=f"مؤشر ستوكاستيك لـ {name} ({symbol})",
            xaxis_title="التاريخ",
            yaxis_title="ستوكاستيك",
            yaxis=dict(range=[0, 100]),
            template='plotly_white', plot_bgcolor='rgba(255,255,255,1)', paper_bgcolor='rgba(255,255,255,1)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        fig.add_annotation(x=x1, y=80, text="ذروة الشراء", showarrow=False, yshift=10, font=dict(color="#e74c3c"))
        fig.add_annotation(x=x1, y=20, text="ذروة البيع", showarrow=False, yshift=-10, font=dict(color="#2ecc71"))

        return fig
    
    def get_technical_analysis(self):
        """
        الحصول على تحليل فني للعملة
        
        العوائد:
            dict: التحليل الفني
        """
        # إضافة المؤشرات الفنية
        df = self.add_all_indicators()
        
        # إذا كانت البيانات فارغة بعد حساب المؤشرات، نعيد تحليلًا افتراضيًا آمنًا
        if df is None or df.empty:
            return {
                'moving_averages': {
                    'trend': '-',
                    'signal': 'انتظار',
                    'details': {
                        'sma7': 0.0,
                        'sma30': 0.0,
                        'sma200': 0.0,
                        'close': 0.0
                    }
                },
                'rsi': {
                    'value': 0.0,
                    'condition': '-',
                    'signal': 'انتظار'
                },
                'macd': {
                    'macd': 0.0,
                    'signal': 0.0,
                    'histogram': 0.0,
                    'condition': '-',
                    'trend_signal': 'انتظار'
                },
                'bollinger_bands': {
                    'close': 0.0,
                    'upper': 0.0,
                    'middle': 0.0,
                    'lower': 0.0,
                    'position': 0.0,
                    'condition': '-',
                    'signal': 'انتظار'
                },
                'stochastic': {
                    'k': 0.0,
                    'd': 0.0,
                    'condition': '-',
                    'signal': 'انتظار'
                },
                'summary': {
                    'overall': 'بيانات غير كافية',
                    'trend': '-',
                    'signal': 'انتظار',
                    'strength': 0,
                    'buy_signals': 0,
                    'sell_signals': 0,
                    'total_indicators': 5
                }
            }

        # الحصول على آخر قيمة
        last_row = df.iloc[-1]
        
        # تحليل المتوسطات المتحركة
        ma_analysis = self._analyze_moving_averages(last_row)
        
        # تحليل مؤشر القوة النسبية
        rsi_analysis = self._analyze_rsi(last_row)
        
        # تحليل مؤشر MACD
        macd_analysis = self._analyze_macd(last_row)
        
        # تحليل نطاقات بولينجر
        bb_analysis = self._analyze_bollinger_bands(last_row)
        
        # تحليل مؤشر ستوكاستيك
        stoch_analysis = self._analyze_stochastic(last_row)
        
        # تجميع التحليل
        analysis = {
            'moving_averages': ma_analysis,
            'rsi': rsi_analysis,
            'macd': macd_analysis,
            'bollinger_bands': bb_analysis,
            'stochastic': stoch_analysis,
            'summary': self._get_analysis_summary(ma_analysis, rsi_analysis, macd_analysis, bb_analysis, stoch_analysis)
        }
        
        return analysis
    
    def _analyze_moving_averages(self, last_row):
        """
        تحليل المتوسطات المتحركة
        
        المعلمات:
            last_row (Series): آخر صف في البيانات
        
        العوائد:
            dict: تحليل المتوسطات المتحركة
        """
        close = last_row['close']
        sma7 = last_row['SMA7']
        sma30 = last_row['SMA30']
        sma200 = last_row['SMA200']
        
        # تحديد الاتجاه
        if close > sma7 > sma30 > sma200:
            trend = 'صعودي قوي'
            signal = 'شراء'
        elif close > sma7 > sma30 and close < sma200:
            trend = 'صعودي متوسط'
            signal = 'شراء بحذر'
        elif close < sma7 < sma30 < sma200:
            trend = 'هبوطي قوي'
            signal = 'بيع'
        elif close < sma7 < sma30 and close > sma200:
            trend = 'هبوطي متوسط'
            signal = 'بيع بحذر'
        elif close > sma7 and close < sma30:
            trend = 'متذبذب قصير المدى'
            signal = 'انتظار'
        else:
            trend = 'متذبذب'
            signal = 'انتظار'
        
        return {
            'trend': trend,
            'signal': signal,
            'details': {
                'close': close,
                'sma7': sma7,
                'sma30': sma30,
                'sma200': sma200
            }
        }
    
    def _analyze_rsi(self, last_row):
        """
        تحليل مؤشر القوة النسبية
        
        المعلمات:
            last_row (Series): آخر صف في البيانات
        
        العوائد:
            dict: تحليل مؤشر القوة النسبية
        """
        rsi = last_row['RSI']
        
        if rsi > 70:
            condition = 'ذروة شراء'
            signal = 'بيع'
        elif rsi < 30:
            condition = 'ذروة بيع'
            signal = 'شراء'
        elif rsi > 50:
            condition = 'إيجابي'
            signal = 'انتظار/شراء'
        else:
            condition = 'سلبي'
            signal = 'انتظار/بيع'
        
        return {
            'value': rsi,
            'condition': condition,
            'signal': signal
        }
    
    def _analyze_macd(self, last_row):
        """
        تحليل مؤشر MACD
        
        المعلمات:
            last_row (Series): آخر صف في البيانات
        
        العوائد:
            dict: تحليل مؤشر MACD
        """
        macd = last_row['MACD']
        signal = last_row['MACD_signal']
        histogram = last_row['MACD_histogram']
        
        if macd > signal and histogram > 0:
            condition = 'إيجابي'
            trend_signal = 'شراء'
        elif macd < signal and histogram < 0:
            condition = 'سلبي'
            trend_signal = 'بيع'
        elif macd > signal and histogram < 0:
            condition = 'تحول إيجابي محتمل'
            trend_signal = 'انتظار/شراء'
        elif macd < signal and histogram > 0:
            condition = 'تحول سلبي محتمل'
            trend_signal = 'انتظار/بيع'
        else:
            condition = 'متذبذب'
            trend_signal = 'انتظار'
        
        return {
            'macd': macd,
            'signal': signal,
            'histogram': histogram,
            'condition': condition,
            'trend_signal': trend_signal
        }
    
    def _analyze_bollinger_bands(self, last_row):
        """
        تحليل نطاقات بولينجر
        
        المعلمات:
            last_row (Series): آخر صف في البيانات
        
        العوائد:
            dict: تحليل نطاقات بولينجر
        """
        close = last_row['close']
        upper = last_row['BB_upper']
        middle = last_row['BB_middle']
        lower = last_row['BB_lower']
        
        # حساب النسبة المئوية للموقع ضمن النطاق
        range_size = upper - lower
        if range_size > 0:
            position = (close - lower) / range_size * 100
        else:
            position = 50
        
        if close > upper:
            condition = 'فوق النطاق العلوي'
            signal = 'بيع/انتظار'
        elif close < lower:
            condition = 'تحت النطاق السفلي'
            signal = 'شراء/انتظار'
        elif position > 80:
            condition = 'قرب النطاق العلوي'
            signal = 'بيع بحذر'
        elif position < 20:
            condition = 'قرب النطاق السفلي'
            signal = 'شراء بحذر'
        else:
            condition = 'ضمن النطاق'
            signal = 'انتظار'
        
        return {
            'close': close,
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'position': position,
            'condition': condition,
            'signal': signal
        }
    
    def _analyze_stochastic(self, last_row):
        """
        تحليل مؤشر ستوكاستيك
        
        المعلمات:
            last_row (Series): آخر صف في البيانات
        
        العوائد:
            dict: تحليل مؤشر ستوكاستيك
        """
        k = last_row['Stoch_K']
        d = last_row['Stoch_D']
        
        if k > 80 and d > 80:
            condition = 'ذروة شراء'
            signal = 'بيع'
        elif k < 20 and d < 20:
            condition = 'ذروة بيع'
            signal = 'شراء'
        elif k > d and k < 80 and d < 80:
            condition = 'إيجابي'
            signal = 'شراء'
        elif k < d and k > 20 and d > 20:
            condition = 'سلبي'
            signal = 'بيع'
        else:
            condition = 'متذبذب'
            signal = 'انتظار'
        
        return {
            'k': k,
            'd': d,
            'condition': condition,
            'signal': signal
        }
    
    def _get_analysis_summary(self, ma, rsi, macd, bb, stoch):
        """
        الحصول على ملخص التحليل
        
        المعلمات:
            ma (dict): تحليل المتوسطات المتحركة
            rsi (dict): تحليل مؤشر القوة النسبية
            macd (dict): تحليل مؤشر MACD
            bb (dict): تحليل نطاقات بولينجر
            stoch (dict): تحليل مؤشر ستوكاستيك
        
        العوائد:
            dict: ملخص التحليل
        """
        # حساب عدد إشارات الشراء والبيع
        buy_signals = 0
        sell_signals = 0
        
        if ma['signal'] in ['شراء', 'شراء بحذر']:
            buy_signals += 1
        elif ma['signal'] in ['بيع', 'بيع بحذر']:
            sell_signals += 1
        
        if rsi['signal'] == 'شراء':
            buy_signals += 1
        elif rsi['signal'] == 'بيع':
            sell_signals += 1
        
        if macd['trend_signal'] in ['شراء', 'انتظار/شراء']:
            buy_signals += 1
        elif macd['trend_signal'] in ['بيع', 'انتظار/بيع']:
            sell_signals += 1
        
        if bb['signal'] in ['شراء/انتظار', 'شراء بحذر']:
            buy_signals += 1
        elif bb['signal'] in ['بيع/انتظار', 'بيع بحذر']:
            sell_signals += 1
        
        if stoch['signal'] == 'شراء':
            buy_signals += 1
        elif stoch['signal'] == 'بيع':
            sell_signals += 1
        
        # تحديد الاتجاه العام
        if buy_signals > sell_signals and buy_signals >= 3:
            trend = 'صعودي'
            strength = (buy_signals - sell_signals) / 5 * 100
            signal = 'شراء'
        elif sell_signals > buy_signals and sell_signals >= 3:
            trend = 'هبوطي'
            strength = (sell_signals - buy_signals) / 5 * 100
            signal = 'بيع'
        else:
            trend = 'متذبذب'
            strength = 50 - abs(buy_signals - sell_signals) * 10
            signal = 'انتظار'
        
        return {
            'trend': trend,
            'strength': strength,
            'signal': signal,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'total_indicators': 5
        }