import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

from flask import Flask, render_template, request
from dotenv import load_dotenv

# تحميل متغيرات البيئة من .env إن وجد
load_dotenv()


def _configure_logging(app: Flask) -> None:
    # ملف السجلات داخل مجلد الحزمة
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]")
    )
    handler.setLevel(logging.INFO)

    app.logger.setLevel(logging.INFO)
    # تجنب إضافة نفس المعالج أكثر من مرة
    if not any(isinstance(h, RotatingFileHandler) for h in app.logger.handlers):
        app.logger.addHandler(handler)


def create_app(config: Optional[dict] = None) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # إعدادات الأمان للجلسة
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-insecure-change-me")
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    if config:
        app.config.update(config)

    _configure_logging(app)

    # تسجيل البلوبرنتات
    # تأخر الاستيراد لتفادي الدوائر
    from .routes import register_blueprints  # noqa: WPS433 (late import for app factory)
    register_blueprints(app)

    # رؤوس أمان أساسية (خفيفة لتفادي كسر Plotly)
    @app.after_request
    def add_security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer-when-downgrade")
        # ملاحظة: يمكن إضافة CSP لاحقًا مع اختبار الواجهات بدقة
        return response

    # معالجات الأخطاء الموحدة
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template("error.html", error_code=404, error_message="الصفحة غير موجودة"), 404

    @app.errorhandler(500)
    def server_error(e):
        app.logger.exception("Unhandled server error: %s", e)
        return render_template("error.html", error_code=500, error_message="خطأ في الخادم"), 500

    # سجل تشغيل أساسي
    app.logger.info("Flask app initialized. Debug=%s", app.debug)

    return app