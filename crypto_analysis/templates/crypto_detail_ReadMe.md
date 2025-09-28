# templates/crypto_detail.html — صفحة تفاصيل العملة

- الهدف: عرض بطاقة معلومات العملة، رسوم تفاعلية (سعر تاريخي، شموع، متوسطات، بولينجر)، وجدول آخر 10 أيام.

## المتغيرات القادمة من routes
- `crypto`: قاموس بيانات العملة المختارة (name, symbol, image, current_price, price_change, market_cap, total_volume, description, data (DataFrame), data_csv اختيارًا).
- `charts`: قاموس JSON لرسوم Plotly التقنية من `TechnicalIndicators.create_technical_charts()` (مثل: `price_ma`, `bollinger`, `rsi`, `macd`, `stochastic`، وربما `volume`).
- `candle_chart`: JSON لمخطط الشموع لآخر 30 يومًا.
- `exchange_rates`: قاموس أسعار الصرف الورقية من `CurrencyConverter`.

## ملاحظات
- تم تبسيط صفحة المعلومات بإزالة بطاقات “التحليل الفني” النصية مع الإبقاء على الرسوم.
- أعمدة CSV الأساسية المطلوبة: `ticker, date, open, high, low, close`، و`volume` اختياري.
- في جدول آخر 10 أيام يتم حساب نسبة التغير مقارنة باليوم السابق داخل الحلقة.

## مواضع الرسم المتوقعة (IDs)
- `historical-price-chart`, `candlestick-chart`, وربما عناصر أخرى مثل `price-ma-chart`, `bollinger-chart` حسب الشيفرة الجافاسكربت داخل الصفحة.
