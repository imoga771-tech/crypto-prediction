# templates/index.html — الصفحة الرئيسية

- الهدف: عرض قائمة العملات مع أسعار حية ومعلومات أساسية وروابط للصفحات.

## المتغيرات القادمة من routes
- `crypto_data`: قاموس الرموز → معلومات (name, image, current_price, price_change, market_cap, total_volume, data/DataFrame).
- `now`: طابع زمني لوقت العرض.

## الميزات
- تحديث أسعار حي (قد يتم عبر نداء `/prices_feed`).
- ترتيب العملات حسب القيمة السوقية.
- روابط إلى تفاصيل كل عملة `/crypto/<symbol>`.

## ملاحظات
- تأكد من وجود حقول `current_price` و`price_change` لتلوين المؤشر (أخضر/أحمر).
- الصور والأسماء قد تأتي من الـ API أو fallback إلى الرمز.
