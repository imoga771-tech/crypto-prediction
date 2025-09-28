import json
import logging
from datetime import datetime
from typing import List

import plotly
import plotly.graph_objects as go
from flask import Blueprint, render_template, request, jsonify

from .services import (
    get_crypto_data,
    get_currency_converter,
    available_symbols,
    clamp_days,
    valid_model_type,
    valid_currency,
    update_prices_in_place,
    get_api_client,
)
from .translations import (
    load_translations,
    build_translations_from_cache,
)
from .models.technical_indicators import TechnicalIndicators
from .models.prediction_models import predict_crypto_price
from .models.crypto_comparison import CryptoComparison

logger = logging.getLogger(__name__)

# Blueprints
main_bp = Blueprint("main", __name__)
detail_bp = Blueprint("detail", __name__)
predict_bp = Blueprint("predict", __name__)
convert_bp = Blueprint("convert", __name__)
compare_bp = Blueprint("compare", __name__)


@main_bp.route("/")
def index():
    # تحديث الأسعار (آمن) لتحسين حداثة البيانات
    try:
        update_prices_in_place()
    except Exception as e:
        logger.error(f"فشل تحديث الأسعار في الصفحة الرئيسية: {e}")

    data = get_crypto_data()

    # ترتيب العملات حسب القيمة السوقية
    sorted_crypto = dict(
        sorted(
            data.items(),
            key=lambda item: item[1].get("market_cap", 0),
            reverse=True,
        )
    )
    return render_template("index.html", crypto_data=sorted_crypto, now=datetime.now())


@main_bp.route("/details")
def details():
    """صفحة تفاصيل المشروع: تعرض نبذة، وإحصائيات البيانات لكل عملة، والتقنيات والمزايا وفريق العمل."""
    data = get_crypto_data()

    # تحضير إحصائيات CSV لكل عملة
    coins_stats = []
    for symbol, meta in data.items():
       # df = meta.get("data_csv") or meta.get("data")
        df = meta.get("data_csv")
        if df is None:
            df = meta.get("data")

        try:
            rows = int(len(df)) if df is not None else 0
            cols = int(len(df.columns)) if df is not None else 0
            date_min = df["date"].min().date().isoformat() if df is not None and "date" in df.columns and not df.empty else None
            date_max = df["date"].max().date().isoformat() if df is not None and "date" in df.columns and not df.empty else None
        except Exception:
            rows, cols, date_min, date_max = 0, 0, None, None

        coins_stats.append({
            "symbol": symbol,
            "name": meta.get("name", symbol),
            "rows": rows,
            "cols": cols,
            "date_min": date_min,
            "date_max": date_max,
        })

    # فرز أبجدي حسب الرمز
    coins_stats = sorted(coins_stats, key=lambda x: x["symbol"])

    # تمرير بيانات إضافية ثابتة للعرض
    tech_stack = [
        "Python (Flask)",
        "Pandas/Numpy",
        "Plotly.js",
        "Bootstrap 5 RTL",
        "jQuery + Select2",
    ]

    models_info = [
        {"key": "linear", "name": "الانحدار الخطي", "desc": "نموذج خطي سريع للأساسيات"},
        {"key": "rf", "name": "الغابات العشوائية", "desc": "تعامل جيد مع العلاقات غير الخطية"},
        {"key": "svr", "name": "SVR", "desc": "ملائم لسلاسل زمنية قصيرة"},
        {"key": "lstm", "name": "LSTM", "desc": "شبكات عصبونية للسلاسل الزمنية"},
        {"key": "ensemble", "name": "مجمع (Ensemble)", "desc": "تجميع مخرجات عدة نماذج"},
    ]

    team = [
        {"name": "ايمن الغيلي", "role": "تطوير الواجهة والباكند"},
        {"name": "مصعب الخطفاء", "role": "تحضير البيانات والنمذجة"},
        {"name": "محمد أمين", "role": "اختبارات وتوثيق"},
        {"name": "حسن آغا", "role": "تحليل فني وتجارب"},
    ]

    supervisor = "صفوان الشيباني"

    return render_template(
        "details.html",
        coins_stats=coins_stats,
        tech_stack=tech_stack,
        models_info=models_info,
        team=team,
        supervisor=supervisor,
    )


@main_bp.route("/update_prices")
def update_prices():
    try:
        summary = update_prices_in_place()
        return jsonify({"success": True, "message": "تم تحديث الأسعار بنجاح", **summary})
    except Exception as e:
        logger.error(f"خطأ في تحديث أسعار العملات: {e}")
        return jsonify({"success": False, "message": f"حدث خطأ: {str(e)}"}), 500


# واجهة JSON لتحديث الأسعار بشكل حي دون إعادة تحميل الصفحة
@main_bp.route("/prices_feed")
def prices_feed():
    try:
        # تحديث الأسعار داخليًا أولاً
        update_prices_in_place()

        data = get_crypto_data()
        payload = {}
        for symbol, meta in data.items():
            payload[symbol] = {
                "current_price": float(meta.get("current_price", 0.0)),
                "price_change": float(meta.get("price_change", 0.0)),
                "market_cap": float(meta.get("market_cap", 0.0)) if meta.get("market_cap") else 0.0,
                "total_volume": float(meta.get("total_volume", 0.0)) if meta.get("total_volume") else 0.0,
                "image": meta.get("image", ""),
                "name": meta.get("name", symbol),
            }
        return jsonify({
            "success": True,
            "data": payload,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"خطأ في prices_feed: {e}")
        return jsonify({"success": False, "message": f"حدث خطأ: {str(e)}"}), 500


@detail_bp.route("/crypto/<symbol>")
def crypto_detail(symbol: str):
    data = get_crypto_data()
    if symbol not in data:
        return "العملة غير موجودة", 404

    # تحديث تفاصيل العملة من API
    try:
        api_client = get_api_client()
        crypto_details = api_client.get_crypto_details(symbol)
        if crypto_details:
            data[symbol]["current_price"] = crypto_details["current_price"]
            data[symbol]["price_change"] = crypto_details["price_change_percentage_24h"]
            data[symbol]["market_cap"] = crypto_details["market_cap"]
            data[symbol]["total_volume"] = crypto_details["total_volume"]
            data[symbol]["last_updated"] = datetime.now()
            if crypto_details.get("description"):
                desc = crypto_details["description"]
                # إذا كان الوصف يحتوي على أحرف عربية، استخدمه؛ غير ذلك احتفظ بالعربي الحالي وخزن الإنجليزي بشكل منفصل
                try:
                    import re
                    has_ar = re.search(r"[\u0600-\u06FF]", str(desc)) is not None
                except Exception:
                    has_ar = False
                if has_ar:
                    data[symbol]["description"] = desc
                else:
                    # لا نستبدل العربي الحالي – نخزن الإنجليزي للاستخدام المستقبلي إن لزم
                    data[symbol]["description_en"] = desc
    except Exception as e:
        logger.error(f"خطأ في تحديث بيانات العملة {symbol}: {e}")

    crypto = data[symbol]

    # تفضيل الوصف العربي من ترجمات الكاش إن توفرت
    try:
        tr_map = load_translations()
        ar_desc = tr_map.get(symbol)
        if ar_desc and isinstance(ar_desc, str) and ar_desc.strip():
            crypto["description"] = ar_desc.strip()
    except Exception as _e:
        pass

    # إنشاء مؤشرات فنية من بيانات CSV فقط (دون دمج API) لضمان التطابق مع الملفات
    source_df = crypto.get("data_csv", crypto["data"])  # fallback إذا لم تتوفر
    tech_indicators = TechnicalIndicators(source_df)

    # إنشاء رسوم بيانية
    charts = tech_indicators.create_technical_charts(symbol, crypto["name"])

    # تحويل الرسوم البيانية إلى JSON
    charts_json = {name: json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) for name, fig in charts.items()}


    # إنشاء رسم بياني للشموع اليابانية (آخر 30 يوم)
    df = source_df
    df_tail = df.iloc[-30:]
    candle_fig = go.Figure(
        data=[
            go.Candlestick(
                x=df_tail["date"],
                open=df_tail["open"],
                high=df_tail["high"],
                low=df_tail["low"],
                close=df_tail["close"],
            )
        ]
    )
    candle_fig.update_layout(
        title=f"مخطط الشموع اليابانية لـ {crypto['name']} (آخر 30 يوم)",
        template="plotly_white",
        plot_bgcolor="rgba(255, 255, 255, 1)",
        paper_bgcolor="rgba(255, 255, 255, 1)",
        xaxis_title="التاريخ",
        yaxis_title="السعر (USD)",
    )
    candle_chart = json.dumps(candle_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # أسعار الصرف
    exchange_rates = get_currency_converter().exchange_rates

    return render_template(
        "crypto_detail.html",
        crypto=crypto,
        charts=charts_json,
        candle_chart=candle_chart,
        exchange_rates=exchange_rates,
    )


@main_bp.route("/admin/build_translations")
def admin_build_translations():
    try:
        summary = build_translations_from_cache()
        return jsonify({"success": True, **summary})
    except Exception as e:
        logger.error(f"ترجمة الأوصاف فشلت: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@predict_bp.route("/predict", methods=["GET", "POST"])
def predict():
    data = get_crypto_data()
    prediction_data = None
    chart_json = None

    model_types = {
        "linear": "الانحدار الخطي",
        "rf": "الغابات العشوائية",
        "svr": "آلة المتجهات الداعمة",
        "lstm": "الشبكات العصبية LSTM",
        "ensemble": "نموذج مجمع (Ensemble)",
    }

    # احصل على العملات الورقية مبكرًا لضمان توفرها حتى في حالات الأخطاء
    try:
        fiat_currencies = get_currency_converter().get_all_fiat_currencies()
    except Exception as e:
        logger.error(f"تعذر جلب العملات الورقية: {e}")
        fiat_currencies = {"USD": "US Dollar"}

    if request.method == "POST":
        symbol = request.form.get("symbol", "").upper().strip()
        days_raw = request.form.get("days", 30)
        currency_raw = request.form.get("currency", "USD").upper().strip()
        model_type_raw = request.form.get("model_type", "ensemble").strip()
        amount_raw = request.form.get("amount", "").strip()

        # تحقق من صحة المدخلات
        valid_symbols = set(available_symbols().keys())
        if symbol not in valid_symbols:
            return render_template(
                "predict.html",
                crypto_data=data,
                model_types=model_types,
                fiat_currencies=fiat_currencies,
                error="رمز العملة غير صالح",
            )

        days = clamp_days(days_raw, 1, 120)
        currency = valid_currency(currency_raw)
        model_type = valid_model_type(model_type_raw)

        try:
            predictions = predict_crypto_price(symbol, days, data, model_type)
            if predictions is None and model_type == 'ensemble':
                # فشل المجمّع: جرّب نماذج فردية تلقائياً
                for fallback in ['linear', 'rf', 'svr']:
                    predictions = predict_crypto_price(symbol, days, data, fallback)
                    if predictions is not None:
                        model_type = fallback
                        break

            if predictions is not None:
                # تحويل الأسعار إلى العملة المطلوبة
                rate = get_currency_converter().exchange_rates.get(currency, 1.0)
                predictions["predicted_price_converted"] = predictions["predicted_price"] * rate

                # إنشاء رسم بياني للتنبؤات
                fig = go.Figure()

                # البيانات التاريخية (آخر 30 يوم)
                historical = data[symbol]["data"].copy().iloc[-30:]
                fig.add_trace(
                    go.Scatter(
                        x=historical["date"],
                        y=historical["close"] * rate,
                        mode="lines",
                        name="البيانات التاريخية",
                        line=dict(color="#3498db", width=2),
                    )
                )

                # التنبؤات
                fig.add_trace(
                    go.Scatter(
                        x=predictions["date"],
                        y=predictions["predicted_price_converted"],
                        mode="lines",
                        name="التنبؤات",
                        line=dict(color="#2ecc71", width=2, dash="dash"),
                    )
                )

                # نطاق ثقة مبسط ±10%
                fig.add_trace(
                    go.Scatter(
                        x=predictions["date"],
                        y=predictions["predicted_price_converted"] * 1.1,
                        mode="lines",
                        name="الحد الأعلى (110%)",
                        line=dict(color="#e74c3c", width=1, dash="dot"),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=predictions["date"],
                        y=predictions["predicted_price_converted"] * 0.9,
                        mode="lines",
                        name="الحد الأدنى (90%)",
                        line=dict(color="#e74c3c", width=1, dash="dot"),
                        fill="tonexty",
                        fillcolor="rgba(231, 76, 60, 0.1)",
                    )
                )

                fig.update_layout(
                    title=f"التنبؤ بسعر {data[symbol]['name']} للـ {days} يوم القادمة بالـ {currency} باستخدام {model_types[model_type]}",
                    template="plotly_white",
                    plot_bgcolor="rgba(255, 255, 255, 1)",
                    paper_bgcolor="rgba(255, 255, 255, 1)",
                    xaxis_title="التاريخ",
                    yaxis_title=f"السعر ({currency})",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )

                chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

                # معلومات شراء افتراضية بناءً على مبلغ اختياري (من الأدوات السريعة في الصفحة الرئيسية)
                purchase_info = None
                try:
                    if amount_raw:
                        amount_val = float(amount_raw)
                        if amount_val > 0:
                            current_price_usd = float(data[symbol]["current_price"])
                            # rate: USD->currency, لذا USD = amount / rate
                            amount_usd = amount_val / (rate if rate else 1.0)
                            units = amount_usd / current_price_usd if current_price_usd else 0.0

                            # إيجاد أقرب تاريخ كان فيه السعر مقاربًا لهذا المبلغ (بالدولار)
                            hist_df = data[symbol]["data"]
                            # تأكد من وجود عمود close
                            if "close" in hist_df.columns and not hist_df.empty:
                                idx = (hist_df["close"] - amount_usd).abs().idxmin()
                                row = hist_df.loc[idx]
                                closest_date = row["date"]
                                closest_price = float(row["close"])
                                diff = float(abs(closest_price - amount_usd))
                            else:
                                closest_date = None
                                closest_price = None
                                diff = None

                            purchase_info = {
                                "input_amount": amount_val,
                                "currency": currency,
                                "amount_usd": round(amount_usd, 6),
                                "units": round(units, 8),
                                "closest_date": closest_date,
                                "closest_price_usd": closest_price,
                                "diff_usd": diff,
                            }
                except Exception:
                    purchase_info = None

                prediction_data = {
                    "symbol": symbol,
                    "name": data[symbol]["name"],
                    "days": days,
                    "currency": currency,
                    "model_type": model_type,
                    "model_name": model_types[model_type],
                    "predictions": predictions.to_dict("records"),
                    "purchase_info": purchase_info,
                }
            else:
                return render_template(
                    "predict.html",
                    crypto_data=data,
                    model_types=model_types,
                    fiat_currencies=fiat_currencies,
                    error="تعذر إجراء التنبؤ لهذه العملة حالياً. جرّب نموذجاً آخر أو عد لاحقاً.",
                )
        except Exception as e:
            logger.error(f"خطأ في التنبؤ بأسعار العملة {symbol}: {e}")
            return render_template(
                "predict.html",
                crypto_data=data,
                model_types=model_types,
                fiat_currencies=fiat_currencies,
                error=f"حدث خطأ أثناء التنبؤ: {str(e)}",
            )

    return render_template(
        "predict.html",
        crypto_data=data,
        prediction_data=prediction_data,
        chart_json=chart_json,
        model_types=model_types,
        fiat_currencies=fiat_currencies,
    )


@convert_bp.route("/convert", methods=["GET", "POST"])
def convert():
    result = None
    popular_conversions = None

    # تحديث أسعار الصرف
    try:
        get_currency_converter().refresh_exchange_rates()
    except Exception as e:
        logger.error(f"خطأ في تحديث أسعار الصرف: {e}")

    # تجهيز أسعار العملات لضمان عدم وجود صفر
    try:
        fiat_rates = {}
        for key, value in get_currency_converter().get_all_fiat_currencies().items():
            try:
                v = float(value)
                fiat_rates[key] = v if v != 0 else 1.0
            except Exception:
                fiat_rates[key] = 1.0
    except Exception as e:
        logger.error(f"خطأ في الحصول على أسعار الصرف: {e}")
        fiat_rates = {}

    if request.method == "POST":
        try:
            amount_raw = request.form.get("amount", "1")
            from_currency = request.form.get("from_currency", "").upper().strip()
            to_currency = request.form.get("to_currency", "").upper().strip()

            # Validate amount
            try:
                amount = float(amount_raw)
                if amount < 0:
                    amount = 0.0
                if amount > 1e12:
                    amount = 1e12
            except Exception:
                amount = 1.0

            converted_amount = get_currency_converter().convert(amount, from_currency, to_currency)

            if converted_amount is not None:
                rate = get_currency_converter().get_conversion_rate(from_currency, to_currency)
                result = {
                    "amount": amount,
                    "from_currency": from_currency,
                    "to_currency": to_currency,
                    "result": converted_amount,
                    "rate": rate,
                }

                popular_conversions = get_currency_converter().get_popular_conversions(from_currency)
        except Exception as e:
            logger.error(f"خطأ في تحويل العملة: {e}")
            return render_template(
                "convert.html",
                currencies={
                    "crypto": {symbol: meta["name"] for symbol, meta in get_crypto_data().items()},
                    "fiat": fiat_rates,
                },
                error=f"حدث خطأ أثناء التحويل: {str(e)}",
            )

    currencies = {
        "crypto": {symbol: meta["name"] for symbol, meta in get_crypto_data().items()},
        "fiat": fiat_rates,
    }

    return render_template(
        "convert.html",
        currencies=currencies,
        result=result,
        popular_conversions=popular_conversions,
        crypto_data=get_crypto_data(),
        current_time=datetime.now(),
    )


@compare_bp.route("/compare", methods=["GET", "POST"])
def compare():
    comparison_charts = None
    comparison_summary = None
    selected_symbols: List[str] = []

    if request.method == "POST":
        symbols_raw = request.form.getlist("symbols") or []
        selected_symbols = [s.upper().strip() for s in symbols_raw]
        valid_syms = set(available_symbols().keys())
        selected_symbols = [s for s in selected_symbols if s in valid_syms]

        days = clamp_days(request.form.get("days", 30), 7, 365)

        if len(selected_symbols) >= 2:
            try:
                comparison = CryptoComparison(get_crypto_data())

                price_chart = comparison.compare_prices(selected_symbols, days)
                volatility_chart = comparison.compare_volatility(selected_symbols, days)
                correlation_chart = comparison.compare_correlation(selected_symbols)
                risk_return_chart = comparison.compare_risk_return(selected_symbols, days)

                comparison_charts = {
                    "price": json.dumps(price_chart, cls=plotly.utils.PlotlyJSONEncoder),
                    "volatility": json.dumps(volatility_chart, cls=plotly.utils.PlotlyJSONEncoder),
                    "correlation": json.dumps(correlation_chart, cls=plotly.utils.PlotlyJSONEncoder),
                    "risk_return": json.dumps(risk_return_chart, cls=plotly.utils.PlotlyJSONEncoder),
                }

                comparison_summary = comparison.get_comparison_summary(selected_symbols, days)
            except Exception as e:
                logger.error(f"خطأ في مقارنة العملات: {e}")
                return render_template(
                    "compare.html",
                    crypto_data=get_crypto_data(),
                    error=f"حدث خطأ أثناء المقارنة: {str(e)}",
                )

    return render_template(
        "compare.html",
        crypto_data=get_crypto_data(),
        comparison_charts=comparison_charts,
        comparison_summary=comparison_summary,
        selected_symbols=selected_symbols,
    )


def register_blueprints(app):
    app.register_blueprint(main_bp)
    app.register_blueprint(detail_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(convert_bp)
    app.register_blueprint(compare_bp)

# ===== Technical analysis route (moved from legacy app.py) =====
@detail_bp.route("/technical/<symbol>")
def technical(symbol: str):
    data = get_crypto_data()
    if symbol not in data:
        return "العملة غير موجودة", 404

    crypto = data[symbol]

    # إنشاء مؤشرات فنية
    tech_indicators = TechnicalIndicators(crypto["data"])

    # إنشاء رسوم بيانية
    charts = tech_indicators.create_technical_charts(symbol, crypto["name"])

    # تحويل الرسوم البيانية إلى JSON
    charts_json = {name: json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) for name, fig in charts.items()}

    # الحصول على التحليل الفني
    technical_analysis = tech_indicators.get_technical_analysis()

    return render_template(
        "technical.html",
        crypto=crypto,
        charts=charts_json,
        technical_analysis=technical_analysis,
    )