import json

# تعریف تنظیمات V2Ray به‌صورت دیکشنری
v2ray_config = {
    "inbounds": [
        {
            "port": 10808,  # پورت ورودی
            "protocol": "vmess",  # پروتکل وی مس
            "settings": {
                "clients": [
                    {
                        "id": "your-uuid-here",  # یک UUID معتبر جایگزین کنید
                        "alterId": 64
                    }
                ]
            },
            "streamSettings": {
                "network": "ws",  # نوع شبکه (می‌توانید tcp, kcp و ... را انتخاب کنید)
                "wsSettings": {
                    "path": "/your-path"  # مسیر وب‌سوکت
                }
            }
        }
    ],
    "outbounds": [
        {
            "protocol": "freedom",
            "settings": {}
        }
    ]
}

# تبدیل دیکشنری به فرمت JSON
config_json = json.dumps(v2ray_config, indent=4)

# ذخیره تنظیمات در فایل config.json
with open("config.json", "w") as file:
    file.write(config_json)

print("فایل تنظیمات V2Ray با موفقیت ایجاد شد: config.json")
