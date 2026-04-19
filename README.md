# Parking Vision Notification Bot

Учебный MVP: Telegram-бот следит за парковками на камерах Ufanet, определяет наличие свободных мест и отправляет уведомление, когда место освобождается.

В проект уже включены:

- обученные веса YOLOv8 в `models/pklot_yolov84_best.pt`;
- артефакты обучения в `pklot_yolov84/`;
- конфиг живых камер Ufanet в `config/cameras.yaml`;
- компактный live-конфиг с самой стабильной камерой в `config/cameras.live_best.yaml`;
- офлайн-демо конфиг в `config/cameras.demo.yaml`, который работает на сохранённых кадрах и подходит для показа преподавателю.

## Возможности

- Получение свежих JPEG-снимков с Ufanet вместо тяжёлого видеопотока.
- Определение состояния парковки по классам `space-empty` и `space-occupied`.
- Хранение подписок и последнего состояния камер в SQLite.
- Подписка в Telegram на конкретную камеру: `24/7` или по временному интервалу, например `18:00-19:00`.
- Офлайн-демо режим с локальными кадрами, чтобы показать бота даже без сети и без ожидания реального освобождения места.

## Структура

- `parking_bot/main.py` - запуск приложения.
- `parking_bot/telegram_bot.py` - Telegram-интерфейс.
- `parking_bot/service.py` - цикл мониторинга и уведомления.
- `parking_bot/detector.py` - обёртка над YOLOv8.
- `parking_bot/camera_catalog.py` - работа с каталогом камер Ufanet и demo-камерами.
- `parking_bot/repository.py` - SQLite-хранилище.
- `config/cameras.yaml` - реальные камеры Ufanet.
- `config/cameras.live_best.yaml` - текущая самая стабильная живая камера для демонстрации.
- `config/cameras.demo.yaml` - офлайн-демо.
- `scripts/check_cameras.py` - разовая проверка камер.
- `scripts/profile_cameras.py` - профилирование камер.
- `scripts/run_self_checks.py` - офлайн self-check без внешних библиотек.

## Установка

1. Установите зависимости:

```powershell
python -m pip install -r requirements.txt
```

2. Создайте `.env`:

```powershell
Copy-Item .env.example .env
```

3. Запишите токен бота в `TELEGRAM_BOT_TOKEN`.

## Запуск

Живой режим с Ufanet-камерами:

```powershell
python -m parking_bot
```

Живой режим только с самой стабильной камерой:

```powershell
python -m parking_bot --config config/cameras.live_best.yaml
```

Офлайн-демо режим:

```powershell
python -m parking_bot --demo
```

Windows helper:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_demo.ps1
```

Linux helper:

```bash
bash scripts/run_demo.sh
```

## Проверка

Быстрая офлайн-проверка базовой логики:

```powershell
python scripts/run_self_checks.py
```

Проверка конфигурации:

```powershell
python scripts/validate_setup.py --demo
python scripts/validate_setup.py
```

Разовая диагностика камер:

```powershell
python scripts/check_cameras.py
python scripts/check_cameras.py --demo
```

Профилирование камер:

```powershell
python scripts/profile_cameras.py --rounds 3 --delay-seconds 5
python scripts/profile_cameras.py --demo --rounds 4 --delay-seconds 1
```

## Сценарий работы бота

1. Пользователь открывает бота и отправляет `/start`.
2. Выбирает камеру.
3. Видит текущий кадр и состояние парковки.
4. Оформляет подписку `24/7` или по временному окну.
5. Когда состояние устойчиво меняется с `full` на `free`, бот отправляет уведомление и актуальный кадр.

## Дообучение модели

Архив тренировки уже распакован в `pklot_yolov84/`. Повторный запуск обучения:

```powershell
python scripts/train_model.py --data path\\to\\data.yaml --epochs 50 --imgsz 640 --batch 16
```

## Raspberry Pi

Для автозапуска можно адаптировать `deploy/systemd/parking-vision-bot.service`, заменить пути и пользователя, затем выполнить:

```bash
sudo systemctl daemon-reload
sudo systemctl enable parking-vision-bot
sudo systemctl start parking-vision-bot
```

## Замечания

- Для живых камер точность зависит от ракурса и освещения; при необходимости модель стоит дообучить на скриншотах конкретных парковок.
- Офлайн-демо не заменяет live-режим, но позволяет гарантированно показать интерфейс, подписку и уведомления.
- Последнее состояние камер хранится в `runtime/parking_bot.db`.
