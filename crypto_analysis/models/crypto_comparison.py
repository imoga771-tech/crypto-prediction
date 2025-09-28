import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

class CryptoComparison:
    """
    فئة لمقارنة العملات الرقمية
    """
    def __init__(self, crypto_data):
        """
        تهيئة المقارنة
        
        المعلمات:
            crypto_data (dict): بيانات العملات الرقمية
        """
        self.crypto_data = crypto_data
    
    def compare_prices(self, symbols, days=30):
        """
        مقارنة أسعار العملات
        
        المعلمات:
            symbols (list): قائمة برموز العملات
            days (int): عدد الأيام
        
        العوائد:
            Figure: الرسم البياني
        """
        if not symbols or len(symbols) < 2:
            return None
        
        fig = go.Figure()
        
        # تحديد تاريخ البداية
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # إضافة خط لكل عملة
        for symbol in symbols:
            if symbol in self.crypto_data:
                df = self.crypto_data[symbol]['data'].copy()
                
                # تصفية البيانات حسب التاريخ
                df['date'] = pd.to_datetime(df['date'])
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                
                # تطبيع الأسعار للمقارنة (النسبة المئوية للتغيير من اليوم الأول)
                first_price = df['close'].iloc[0]
                df['normalized_price'] = (df['close'] / first_price - 1) * 100
                
                # إضافة خط للرسم البياني
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['normalized_price'],
                    mode='lines',
                    name=f"{self.crypto_data[symbol]['name']} ({symbol})",
                    line=dict(width=2)
                ))
        
        # تحديث تخطيط الرسم البياني
        fig.update_layout(
            title="مقارنة أداء العملات الرقمية (نسبة التغيير)",
            xaxis_title="التاريخ",
            yaxis_title="نسبة التغيير (%)",
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
        
        # إضافة خط الصفر
        fig.add_shape(
            type="line",
            x0=start_date,
            y0=0,
            x1=end_date,
            y1=0,
            line=dict(color="gray", width=1, dash="dash")
        )
        
        return fig
    
    def compare_volatility(self, symbols, days=30):
        """
        مقارنة تقلب العملات
        
        المعلمات:
            symbols (list): قائمة برموز العملات
            days (int): عدد الأيام
        
        العوائد:
            Figure: الرسم البياني
        """
        if not symbols or len(symbols) < 2:
            return None
        
        # حساب التقلب لكل عملة
        volatility_data = []
        
        for symbol in symbols:
            if symbol in self.crypto_data:
                df = self.crypto_data[symbol]['data'].copy()
                
                # تصفية البيانات حسب التاريخ
                df['date'] = pd.to_datetime(df['date'])
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                
                # حساب التقلب اليومي
                df['daily_return'] = df['close'].pct_change() * 100
                
                # حساب الانحراف المعياري للعوائد اليومية
                volatility = df['daily_return'].std()
                
                volatility_data.append({
                    'symbol': symbol,
                    'name': self.crypto_data[symbol]['name'],
                    'volatility': volatility
                })
        
        # ترتيب البيانات حسب التقلب
        volatility_data = sorted(volatility_data, key=lambda x: x['volatility'], reverse=True)
        
        # إنشاء الرسم البياني
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=[f"{item['name']} ({item['symbol']})" for item in volatility_data],
            y=[item['volatility'] for item in volatility_data],
            marker_color='#3498db'
        ))
        
        # تحديث تخطيط الرسم البياني
        fig.update_layout(
            title=f"مقارنة تقلب العملات الرقمية (آخر {days} يوم)",
            xaxis_title="العملة",
            yaxis_title="التقلب (الانحراف المعياري للعوائد اليومية %)",
            template='plotly_white',
            plot_bgcolor='rgba(255, 255, 255, 1)',
            paper_bgcolor='rgba(255, 255, 255, 1)'
        )
        
        return fig
    
    def compare_correlation(self, symbols):
        """
        مقارنة الارتباط بين العملات
        
        المعلمات:
            symbols (list): قائمة برموز العملات
        
        العوائد:
            Figure: الرسم البياني
        """
        if not symbols or len(symbols) < 2:
            return None
        
        # جمع بيانات الأسعار لجميع العملات
        price_data = {}
        
        for symbol in symbols:
            if symbol in self.crypto_data:
                df = self.crypto_data[symbol]['data'].copy()
                df['date'] = pd.to_datetime(df['date'])
                price_data[symbol] = df[['date', 'close']].rename(columns={'close': symbol})
        
        # دمج البيانات
        merged_data = None
        
        for symbol, df in price_data.items():
            if merged_data is None:
                merged_data = df
            else:
                merged_data = pd.merge(merged_data, df, on='date', how='inner')
        
        if merged_data is None or merged_data.shape[1] <= 2:
            return None
        
        # حساب العوائد اليومية
        returns_data = merged_data.set_index('date')
        returns_data = returns_data.pct_change().dropna()
        
        # حساب مصفوفة الارتباط
        correlation_matrix = returns_data.corr()
        
        # إنشاء الرسم البياني
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
            z=correlation_matrix.values,
            x=[f"{self.crypto_data[symbol]['name']} ({symbol})" for symbol in correlation_matrix.columns],
            y=[f"{self.crypto_data[symbol]['name']} ({symbol})" for symbol in correlation_matrix.index],
            colorscale='RdBu',
            zmid=0,
            text=np.around(correlation_matrix.values, decimals=2),
            texttemplate='%{text}',
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        # تحديث تخطيط الرسم البياني
        fig.update_layout(
            title="مصفوفة الارتباط بين العملات الرقمية",
            template='plotly_white',
            plot_bgcolor='rgba(255, 255, 255, 1)',
            paper_bgcolor='rgba(255, 255, 255, 1)'
        )
        
        return fig
    
    def compare_risk_return(self, symbols, days=30):
        """
        مقارنة المخاطر والعوائد
        
        المعلمات:
            symbols (list): قائمة برموز العملات
            days (int): عدد الأيام
        
        العوائد:
            Figure: الرسم البياني
        """
        if not symbols or len(symbols) < 2:
            return None
        
        # حساب المخاطر والعوائد لكل عملة
        risk_return_data = []
        
        for symbol in symbols:
            if symbol in self.crypto_data:
                df = self.crypto_data[symbol]['data'].copy()
                
                # تصفية البيانات حسب التاريخ
                df['date'] = pd.to_datetime(df['date'])
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                
                # حساب العوائد اليومية
                df['daily_return'] = df['close'].pct_change() * 100
                
                # حساب متوسط العائد اليومي
                avg_return = df['daily_return'].mean()
                
                # حساب الانحراف المعياري (المخاطر)
                risk = df['daily_return'].std()
                
                # حساب العائد الإجمالي
                total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
                
                risk_return_data.append({
                    'symbol': symbol,
                    'name': self.crypto_data[symbol]['name'],
                    'avg_return': avg_return,
                    'risk': risk,
                    'total_return': total_return,
                    'sharpe_ratio': avg_return / risk if risk > 0 else 0
                })
        
        # إنشاء الرسم البياني
        fig = go.Figure()
        
        for item in risk_return_data:
            fig.add_trace(go.Scatter(
                x=[item['risk']],
                y=[item['avg_return']],
                mode='markers+text',
                marker=dict(
                    size=abs(item['total_return']) / 2 + 10,
                    color='#3498db' if item['total_return'] >= 0 else '#e74c3c',
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                text=f"{item['name']} ({item['symbol']})",
                textposition="top center",
                name=f"{item['name']} ({item['symbol']})"
            ))
        
        # تحديث تخطيط الرسم البياني
        fig.update_layout(
            title=f"مقارنة المخاطر والعوائد للعملات الرقمية (آخر {days} يوم)",
            xaxis_title="المخاطر (الانحراف المعياري للعوائد اليومية %)",
            yaxis_title="متوسط العائد اليومي (%)",
            template='plotly_white',
            plot_bgcolor='rgba(255, 255, 255, 1)',
            paper_bgcolor='rgba(255, 255, 255, 1)',
            showlegend=False
        )
        
        # إضافة خطوط الأصل
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=max([item['risk'] for item in risk_return_data]) * 1.1,
            y1=0,
            line=dict(color="gray", width=1, dash="dash")
        )
        
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=0,
            y1=max([abs(item['avg_return']) for item in risk_return_data]) * 1.1,
            line=dict(color="gray", width=1, dash="dash")
        )
        
        return fig
    
    def get_comparison_summary(self, symbols, days=30):
        """
        الحصول على ملخص المقارنة
        
        المعلمات:
            symbols (list): قائمة برموز العملات
            days (int): عدد الأيام
        
        العوائد:
            dict: ملخص المقارنة
        """
        if not symbols or len(symbols) < 2:
            return None
        
        # جمع البيانات لكل عملة
        comparison_data = []
        
        for symbol in symbols:
            if symbol in self.crypto_data:
                df = self.crypto_data[symbol]['data'].copy()
                
                # تصفية البيانات حسب التاريخ
                df['date'] = pd.to_datetime(df['date'])
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                
                # حساب العوائد اليومية
                df['daily_return'] = df['close'].pct_change() * 100
                
                # حساب المؤشرات
                current_price = df['close'].iloc[-1]
                price_change = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
                avg_daily_return = df['daily_return'].mean()
                volatility = df['daily_return'].std()
                max_drawdown = self._calculate_max_drawdown(df['close'])
                sharpe_ratio = avg_daily_return / volatility if volatility > 0 else 0
                
                comparison_data.append({
                    'symbol': symbol,
                    'name': self.crypto_data[symbol]['name'],
                    'current_price': current_price,
                    'price_change': price_change,
                    'avg_daily_return': avg_daily_return,
                    'volatility': volatility,
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': sharpe_ratio
                })
        
        # ترتيب البيانات حسب العائد
        comparison_data = sorted(comparison_data, key=lambda x: x['price_change'], reverse=True)
        
        # تحديد أفضل وأسوأ أداء
        best_performer = comparison_data[0] if comparison_data else None
        worst_performer = comparison_data[-1] if comparison_data else None
        
        # تحديد الأقل تقلبًا
        least_volatile = min(comparison_data, key=lambda x: x['volatility']) if comparison_data else None
        
        # تحديد الأفضل من حيث نسبة شارب
        best_sharpe = max(comparison_data, key=lambda x: x['sharpe_ratio']) if comparison_data else None
        
        return {
            'comparison_data': comparison_data,
            'best_performer': best_performer,
            'worst_performer': worst_performer,
            'least_volatile': least_volatile,
            'best_sharpe': best_sharpe,
            'period': days
        }
    
    def _calculate_max_drawdown(self, prices):
        """
        حساب أقصى انخفاض
        
        المعلمات:
            prices (Series): سلسلة الأسعار
        
        العوائد:
            float: أقصى انخفاض
        """
        # حساب أقصى انخفاض
        peak = prices.expanding().max()
        drawdown = (prices / peak - 1) * 100
        max_drawdown = drawdown.min()
        
        return max_drawdown