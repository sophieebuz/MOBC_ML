# Домашнее задание №1
В репозитории модно найти следющие файлы:
 - `a`


Этапы проекта:
 - простейший EDA
   * визаульный анализ датасетов, анализ описательных статистик
   * анализ датасета на наличие дубликатов, обработка их
   * убрать единицы измерения для признаков mileage, engine, max_power и сделать столбцы числовыми
   * обработка столбца torque (см. подробности обработки в ноутбуке): разделение его на 'torque' и 'max_torque_rp
   * заполнение пропусков в данных медианами (пропуски встречаются только в числовых признаках)
 - визуализация данных
   * построение попарных распределений и гистограмм числовых признаков
   * построение матрицы корреляций
 - обучение модели только на вещественных признаках 
   * построенные модели: линейная регрессия, Lasso регрессии, ElasticNet
   * подбор гиперпараметров с помощью поиска по сетке
   * построение на неотмасштабируемых и отмасштабируемых признаках  
   Все модели друг от друга не особо отличались по качеству.
 - обучение модели только на вещественных и категориальных признаках 
   * кодирование категориальных признаках с помощью one-hot encoding
   * построенные модели: Ridge
   * подбор гиперпараметров с помощью поиска по сетке
   * построение на неотмасштабируемых и отмасштабируемых вещественных признаках  
   Качество у моделей хоть и стало получше, но не особо сильно выросло.
 - feature engeneering
   * добавление новых столбцов:
      - квадрат года
      - пробег автомобиля за год
      - расчет числа "лошадей" на литр объема
      - название фирмы, которая произвела автомобиль
      - создание порогового столбца "1вый владелец+test drive car" и "2ой и более владелец"  
   * построение моделей на датасетах с добавленными признакамиЖ
      - модель Ridge
      - построение на неотмасштабируемых и отмасштабируемых признаках
      - подбор гиперпараметров с помощью поиска по сетке

Добавление новых сгенерированных признаков дало прирост в качестве. Однако поскольку все они добавились в модель одновременно, нельзя сказать. какой из них конкретно привнес в модель наибольший вклад в качество  предсказний.  

 - расчет бизнесовой метрики качества
   * расчет производился на 3х моделях (Ridge регрессия на отмасштабируемых/неотмасштабируемых данных на датасетах с/без признаками, полученными в feature engeneering)
   
Итоговая модель (модель, которая показала наилучшее качество) приняла следующий вид: Ridge регрессия с применением onehot encoding для катеориальных признаков, вещественные признаки брались неотмасштабированными (бизнес метрика на неотмасштабированных показала лучшее качество) + добавление признаков, сгенерированным в feature engeneering-е

В целом можно добавить еще, что в принципе на этих данных с помощью линейной регрессии супер хорошего качества не добиться (особенно на бизнесс метрики, тк единицы измерения цены автомобилей в млн и ошибки очень велики).

 - реализация серсива на FastApi по предсказанию цены машин/ы (см. коментарии в ноутбуке)
   * предсказание на одном обекте:
      - объект подается без пропусков в данных
![img1](https://github.com/sophieebuz/MOBC_ML/blob/main/screenshots/Screenshot_3.jpg)
![img2](https://github.com/sophieebuz/MOBC_ML/blob/main/screenshots/Screenshot_4.jpg)
      - объект подается c пропусками в данных (в числовых столбцах)
![img1](https://github.com/sophieebuz/MOBC_ML/blob/main/screenshots/Screenshot_5.jpg)
![img2](https://github.com/sophieebuz/MOBC_ML/blob/main/screenshots/Screenshot_6.jpg)
   * предсказание на нескольких обектах:
      - объект подается без пропусков в данных
![img1](https://github.com/sophieebuz/MOBC_ML/blob/main/screenshots/Screenshot_7.jpg)
![img2](https://github.com/sophieebuz/MOBC_ML/blob/main/screenshots/Screenshot_8.jpg)
![img3](https://github.com/sophieebuz/MOBC_ML/blob/main/screenshots/Screenshot_9.jpg)
      - объект подается c пропусками в данных (в числовых столбцах)
![img1](https://github.com/sophieebuz/MOBC_ML/blob/main/screenshots/Screenshot_1.jpg)
![img2](https://github.com/sophieebuz/MOBC_ML/blob/main/screenshots/Screenshot_2.jpg)

В сервере хотелось бы еще добавить пользоватеьсий интерфкйс (чтобы модно было в него загрузить данные по объекту/ам и получить туда же предсказание/я), но на это времени не хватило.





  
   
