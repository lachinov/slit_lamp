# Сегментация капилляров глаза человека по снимкам с офтальмологической щелевой лампы

Docker находится в `scripts/Dockerfiles/Dockerfile`

# Структура папок

```
├── eye_test
├── eye_train
├──── masks
├── models
├── scripts
├── splits
├── unet_deploy_flask
```

`eye_test` папка с тестовыми файлами

`eye_train` папка с тренировычными изображениями и geojson разметкой

`eye_train/masks` папка с png разметкой

`models` папка с моделями

`scripts` тренировочные скрипты

`splits` сплиты для 5fold cv

`unet_deploy_flask` webapp

Рабочая директория - `scripts`

# Предобработка

запустить `make_image_dataset.py`

# Конфигурации

находятся в `scripts/configs`. Суффикс `.used` означает использованные. Для запуска нужно удалить этот суффикс. Скрипт `main_config_queue.py` берет первый конфиг с суффиксом `.json` и исполняет его.

# Тренировка

- Поместите конфиги в папку `scripts/configs`.
- запустите `main_config_queue.py`

# Валидация

Пример

```
validate.py --config slit_lamp/models/test_006_u_b4_depth3_adam_long_conv.json/config.json
--validation_path
slit_lamp/eye_train
```

# Предсказание

Пример
```
predict.py --output_path slit_lamp/tmp/submission --test_path slit_lamp/eye_test --config slit_lamp/models/test_006_u_b4_depth3_adam_long_conv.json/config.json
```


