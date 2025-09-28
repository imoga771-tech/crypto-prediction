import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os

class PredictionModel:
    """
    فئة أساسية لنماذج التنبؤ بأسعار العملات الرقمية
    """
    def __init__(self, data, model_type='linear'):
        """
        تهيئة نموذج التنبؤ
        
        المعلمات:
            data (DataFrame): بيانات العملة التاريخية
            model_type (str): نوع النموذج ('linear', 'rf', 'svr', 'lstm')
        """
        self.data = data.copy()
        self.model_type = model_type
        self.model = None
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.features = None
        self.target = None
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
        
        # إنشاء مجلد لحفظ النماذج إذا لم يكن موجودًا
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
    
    def prepare_data(self, window_size=30, target_column='close'):
        """
        إعداد البيانات للتدريب
        
        المعلمات:
            window_size (int): حجم النافذة للبيانات التاريخية
            target_column (str): عمود الهدف للتنبؤ
        """
        # إضافة مؤشرات فنية
        self._add_technical_indicators()
        
        # إعداد البيانات حسب نوع النموذج
        if self.model_type == 'lstm':
            self._prepare_lstm_data(window_size, target_column)
        else:
            self._prepare_standard_data(target_column)
    
    def _add_technical_indicators(self):
        """إضافة مؤشرات فنية للبيانات"""
        df = self.data
        
        # إضافة المتوسطات المتحركة
        df['MA7'] = df['close'].rolling(window=7).mean()
        df['MA14'] = df['close'].rolling(window=14).mean()
        df['MA30'] = df['close'].rolling(window=30).mean()
        
        # إضافة مؤشر القوة النسبية (RSI)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # إضافة مؤشر تقارب وتباعد المتوسطات المتحركة (MACD)
        df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # إضافة نطاقات بولينجر
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['20STD'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['MA20'] + (df['20STD'] * 2)
        df['lower_band'] = df['MA20'] - (df['20STD'] * 2)
        
        # إضافة مؤشر تدفق الأموال (MFI)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume'] if 'volume' in df.columns else typical_price
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).fillna(0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).fillna(0)
        
        positive_mf = positive_flow.rolling(window=14).sum()
        negative_mf = negative_flow.rolling(window=14).sum()
        
        mfi_ratio = positive_mf / negative_mf
        df['MFI'] = 100 - (100 / (1 + mfi_ratio))
        
        # حذف الصفوف التي تحتوي على قيم NaN
        self.data = df.dropna().reset_index(drop=True)
    
    def _prepare_standard_data(self, target_column):
        """إعداد البيانات للنماذج القياسية"""
        df = self.data
        
        # إضافة ميزة الأيام من البداية
        df['days_from_start'] = (df['date'] - df['date'].min()).dt.days
        
        # تحديد الميزات والهدف
        feature_columns = ['days_from_start', 'MA7', 'MA14', 'MA30', 'RSI', 'MACD', 'MACD_signal', 
                          'upper_band', 'lower_band', 'MFI']
        
        # التحقق من وجود عمود الحجم
        if 'volume' in df.columns:
            feature_columns.append('volume')
        
        self.features = df[feature_columns].values
        self.target = df[[target_column]].values
        
        # تطبيع البيانات
        self.features = self.scaler_x.fit_transform(self.features)
        self.target = self.scaler_y.fit_transform(self.target)
    
    def _prepare_lstm_data(self, window_size, target_column):
        """إعداد البيانات لنموذج LSTM"""
        df = self.data

        # تحديد الميزات والهدف
        feature_columns = ['open', 'high', 'low', 'close', 'MA7', 'MA14', 'MA30', 'RSI', 'MACD',
                           'MACD_signal', 'upper_band', 'lower_band', 'MFI']

        # التحقق من وجود عمود الحجم
        if 'volume' in df.columns:
            feature_columns.append('volume')

        # حفظ قائمة ميزات LSTM للاستخدام أثناء التنبؤ
        self.lstm_feature_columns = feature_columns.copy()

        # تطبيع البيانات
        data_scaled = self.scaler_x.fit_transform(df[feature_columns])

        # تحقق من كفاية البيانات لبناء تسلسلات LSTM
        if (len(data_scaled) - window_size) <= 0:
            raise ValueError("Insufficient data for LSTM window")

        # إنشاء مجموعات البيانات المتسلسلة
        X, y = [], []
        for i in range(len(data_scaled) - window_size):
            X.append(data_scaled[i:i + window_size])
            y.append(df[target_column].iloc[i + window_size])

        self.features = np.array(X)
        self.target = np.array(y).reshape(-1, 1)

        # تطبيع الهدف
        self.target = self.scaler_y.fit_transform(self.target)
    
    def train(self):
        """تدريب النموذج"""
        if self.features is None or self.target is None:
            raise ValueError("يجب إعداد البيانات أولاً باستخدام prepare_data()")
        
        # تقسيم البيانات إلى مجموعات تدريب واختبار
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42
        )
        
        # تدريب النموذج حسب النوع
        if self.model_type == 'linear':
            self.model = LinearRegression()
            self.model.fit(X_train, y_train)
        
        elif self.model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train.ravel())
        
        elif self.model_type == 'svr':
            self.model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
            self.model.fit(X_train, y_train.ravel())
        
        elif self.model_type == 'lstm':
            # إنشاء نموذج LSTM
            self.model = Sequential()
            self.model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
            self.model.add(Dropout(0.2))
            self.model.add(LSTM(units=50, return_sequences=False))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(units=25))
            self.model.add(Dense(units=1))
            
            # تجميع النموذج
            self.model.compile(optimizer='adam', loss='mean_squared_error')
            
            # تدريب النموذج
            self.model.fit(
                X_train, y_train, 
                epochs=50, 
                batch_size=32, 
                validation_data=(X_test, y_test),
                verbose=0
            )
    
    def save_model(self, symbol):
        """
        حفظ النموذج
        
        المعلمات:
            symbol (str): رمز العملة
        """
        if self.model is None:
            raise ValueError("يجب تدريب النموذج أولاً")
        
        model_filename = f"{symbol}_{self.model_type}_model"
        
        if self.model_type == 'lstm':
            # حفظ نموذج LSTM
            model_path = os.path.join(self.model_path, model_filename)
            self.model.save(model_path)
        else:
            # حفظ النماذج الأخرى
            model_path = os.path.join(self.model_path, f"{model_filename}.joblib")
            joblib.dump(self.model, model_path)
        
        # حفظ المقاييس
        scaler_x_path = os.path.join(self.model_path, f"{symbol}_{self.model_type}_scaler_x.joblib")
        scaler_y_path = os.path.join(self.model_path, f"{symbol}_{self.model_type}_scaler_y.joblib")
        
        joblib.dump(self.scaler_x, scaler_x_path)
        joblib.dump(self.scaler_y, scaler_y_path)
    
    def load_model(self, symbol):
        """
        تحميل النموذج
        
        المعلمات:
            symbol (str): رمز العملة
        
        العوائد:
            bool: True إذا تم تحميل النموذج بنجاح، False خلاف ذلك
        """
        model_filename = f"{symbol}_{self.model_type}_model"
        
        try:
            if self.model_type == 'lstm':
                # تحميل نموذج LSTM
                model_path = os.path.join(self.model_path, model_filename)
                self.model = tf.keras.models.load_model(model_path)
            else:
                # تحميل النماذج الأخرى
                model_path = os.path.join(self.model_path, f"{model_filename}.joblib")
                self.model = joblib.load(model_path)
            
            # تحميل المقاييس
            scaler_x_path = os.path.join(self.model_path, f"{symbol}_{self.model_type}_scaler_x.joblib")
            scaler_y_path = os.path.join(self.model_path, f"{symbol}_{self.model_type}_scaler_y.joblib")
            
            self.scaler_x = joblib.load(scaler_x_path)
            self.scaler_y = joblib.load(scaler_y_path)
            
            return True
        except Exception:
            # أي خطأ أثناء التحميل يعتبر كأن النموذج غير موجود/فاسد
            # ليتم لاحقًا إعادة التدريب وحفظ نموذج صالح
            return False
    
    def predict(self, days, last_data=None):
        """
        التنبؤ بالأسعار المستقبلية
        
        المعلمات:
            days (int): عدد الأيام للتنبؤ
            last_data (DataFrame): آخر البيانات المتاحة (للنماذج المتقدمة)
        
        العوائد:
            DataFrame: إطار بيانات يحتوي على التنبؤات
        """
        if self.model is None:
            raise ValueError("يجب تدريب النموذج أولاً أو تحميله")
        
        # الحصول على آخر تاريخ في البيانات
        last_date = self.data['date'].max()
        
        # إنشاء تواريخ للتنبؤات
        future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        
        if self.model_type == 'linear':
            # التنبؤ باستخدام نموذج الانحدار الخطي
            last_day = (self.data['date'].max() - self.data['date'].min()).days
            future_days = np.array([[last_day + i + 1] for i in range(days)])

            # إعداد البيانات للتنبؤ:
            # استخدم آخر متجه ميزات كأساس، وغيّر days_from_start لكل يوم
            last_feat = self.features[-1].copy()
            future_features = np.tile(last_feat, (days, 1))
            future_features[:, 0] = future_days.flatten()

            # التنبؤ
            predictions = self.model.predict(future_features)
            predictions = self.scaler_y.inverse_transform(predictions)
            
        elif self.model_type in ['rf', 'svr']:
            # التنبؤ باستخدام النماذج المتقدمة
            predictions = []
            current_features = self.features[-1:].copy()
            
            for i in range(days):
                # التنبؤ باليوم التالي
                pred = self.model.predict(current_features)
                predictions.append(pred[0])
                
                # تحديث الميزات للتنبؤ التالي
                current_features[0, 0] += 1  # زيادة الأيام
                
                # يمكن تحديث المؤشرات الفنية هنا للحصول على تنبؤات أكثر دقة
            
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler_y.inverse_transform(predictions)
            
        elif self.model_type == 'lstm':
            # التنبؤ باستخدام نموذج LSTM
            predictions = []
            current_sequence = self.features[-1:].copy()  # الشكل: (1, window, n_features)

            # موقع عمود الإغلاق ضمن ميزات LSTM
            close_idx = None
            if hasattr(self, "lstm_feature_columns") and "close" in self.lstm_feature_columns:
                close_idx = self.lstm_feature_columns.index("close")

            for i in range(days):
                # التنبؤ باليوم التالي (مُقاس بمقياس y)
                pred_scaled_y = self.model.predict(current_sequence, verbose=0)[0, 0]
                predictions.append(pred_scaled_y)

                # إذا توفرت معلومات الأعمدة والمقاييس، حدّث آخر خطوة في التسلسل بقيمة إغلاق جديدة
                if close_idx is not None and hasattr(self.scaler_x, "scale_") and hasattr(self.scaler_x, "min_"):
                    # تحويل التنبؤ إلى القيمة الحقيقية (غير مُقاسة)
                    pred_unscaled = self.scaler_y.inverse_transform(np.array([[pred_scaled_y]])).ravel()[0]
                    # تحويل القيمة الحقيقية للإغلاق إلى الفضاء المُقاس بمقياس X
                    new_close_scaled_for_x = pred_unscaled * self.scaler_x.scale_[close_idx] + self.scaler_x.min_[close_idx]

                    # بناء خطوة جديدة بالاعتماد على آخر خطوة مع تعديل الإغلاق فقط
                    last_step = current_sequence[:, -1, :].copy()  # (1, n_features)
                    last_step[0, close_idx] = new_close_scaled_for_x

                    # إزاحة النافذة وإلحاق الخطوة الجديدة
                    current_sequence = np.concatenate(
                        [current_sequence[:, 1:, :], last_step.reshape(1, 1, -1)],
                        axis=1
                    )
                else:
                    # في حال عدم توفر المعلومات الكافية، نُبقي على الإزاحة فقط لتفادي كسر الأبعاد
                    current_sequence = np.concatenate(
                        [current_sequence[:, 1:, :], current_sequence[:, -1:, :]],
                        axis=1
                    )

            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler_y.inverse_transform(predictions)
        
        # إنشاء DataFrame للتنبؤات
        predictions_df = pd.DataFrame({
            'date': future_dates,
            'predicted_price': predictions.flatten()
        })
        
        return predictions_df


def predict_crypto_price(symbol, days, crypto_data, model_type='ensemble'):
    """
    التنبؤ بأسعار العملات الرقمية باستخدام مجموعة من النماذج
    
    المعلمات:
        symbol (str): رمز العملة
        days (int): عدد الأيام للتنبؤ
        crypto_data (dict): بيانات العملات الرقمية
        model_type (str): نوع النموذج ('linear', 'rf', 'svr', 'lstm', 'ensemble')
    
    العوائد:
        DataFrame: إطار بيانات يحتوي على التنبؤات
    """
    if symbol not in crypto_data:
        return None
    
    df = crypto_data[symbol]['data'].copy()
    
    if model_type == 'ensemble':
        # استخدام مجموعة من النماذج للتنبؤ
        models = ['linear', 'rf', 'svr']
        predictions_list = []
        
        for model_name in models:
            model = PredictionModel(df, model_type=model_name)
            loaded = model.load_model(symbol)
            try:
                # إعداد البيانات دائمًا لضمان توفر self.features حتى عند تحميل النموذج
                model.prepare_data()
                if not loaded:
                    model.train()
                    model.save_model(symbol)
                preds = model.predict(days)
                predictions_list.append(preds['predicted_price'].values)
            except Exception:
                # تخطي النموذج الذي يفشل لأي سبب (بيانات غير كافية/مشاكل أخرى)
                continue
         
        # إذا لم ينجح أي نموذج، أعد None ليتم التعامل معه في الطبقة الأعلى
        if len(predictions_list) == 0:
            return None
        # حساب متوسط التنبؤات
        ensemble_predictions = np.mean(predictions_list, axis=0)
        
        # إنشاء DataFrame للتنبؤات
        last_date = df['date'].max()
        future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        
        predictions_df = pd.DataFrame({
            'date': future_dates,
            'predicted_price': ensemble_predictions
        })
        
        return predictions_df
    else:
        # استخدام نموذج واحد للتنبؤ
        primary_type = model_type
        model = PredictionModel(df, model_type=primary_type)
        loaded = model.load_model(symbol)
        try:
            model.prepare_data()
            if not loaded:
                model.train()
                model.save_model(symbol)
            return model.predict(days)
        except Exception:
            # fallback ذكي إذا فشل LSTM أو أي نموذج بسبب البيانات
            fallback_order = ['rf', 'linear'] if primary_type == 'lstm' else (['linear'] if primary_type == 'rf' else ['rf', 'linear'])
            for fb in fallback_order:
                try:
                    fb_model = PredictionModel(df, model_type=fb)
                    fb_loaded = fb_model.load_model(symbol)
                    fb_model.prepare_data()
                    if not fb_loaded:
                        fb_model.train()
                        fb_model.save_model(symbol)
                    return fb_model.predict(days)
                except Exception:
                    continue
            # إذا فشل كل شيء، أعد None ليتم التعامل معه أعلى
            return None