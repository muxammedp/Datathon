SOURCE = {
    'document': 'Постановление Правления НБРК от 29.11.2019 № 231',
    'url': 'https://adilet.zan.kz/rus/docs/V1900019680',
    'reading_url': 'https://zakon.uchet.kz/rus/docs/V1900019680',
    'checked_on': '2026-10-17',
    'edition_note': 'Текст сверен по публичной копии с изменениями на 12.07.2026. Перед внедрением требуется сверка официальной редакции.',
}

RULES = {
    'camera_phone': {
        'title': 'Смартфон в кассовой зоне',
        'section': 'Пункт 12',
        'rule': 'В кассовый узел допускаются средства связи без функции фото- и видеофиксации.',
        'criterion': 'The CASHIER clearly uses or stores a handheld smartphone in the cash-handling workspace, with a visible camera module or visible photo/video capture. Distinguish it from a calculator, payment terminal, document, banknote, desk telephone, or a customer phone outside the cash area. Do not infer a camera function from an indistinct rectangle.',
        'limitation': 'Нужно отличить смартфон от терминала и подтвердить функцию съёмки; средства связи без камеры допускаются.',
    },
    'personal_belongings': {
        'title': 'Личные вещи в кассовой зоне',
        'section': 'Пункт 12',
        'rule': 'Личные вещи работников хранятся вне кассового узла; для медицинских средств предусмотрено исключение.',
        'criterion': 'An unmistakable personal handbag, backpack or wallet is placed or opened by the cashier inside their cash-handling workspace. Exclude client belongings on the client side, collection bags, document folders, work equipment, medicine and medical devices. Do not guess ownership of an ambiguous bag.',
        'limitation': 'Вещь должна быть явно личной и находиться на стороне кассира, а не клиента; инкассаторская сумка не относится к этой категории.',
    },
    'cashier_left': {
        'title': 'Уход кассира во время обслуживания',
        'section': 'Пункт 22',
        'rule': 'Кассир не должен уходить до окончания обслуживания клиента.',
        'criterion': 'The cashier walks away from the workstation while the customer is still waiting and the cash transaction is visibly unfinished. Standing up or reaching for nearby equipment is not leaving.',
        'limitation': 'По видео должно быть видно, что обслуживание ещё продолжается.',
    },
    'cash_storage': {
        'title': 'Возможное нарушение хранения наличных',
        'section': 'Пункт 20',
        'rule': 'Ранее принятые деньги хранятся в предназначенном для этого запираемом устройстве.',
        'criterion': 'Cash from a completed previous customer transaction remains openly on the desk when another transaction begins, or the cashier clearly puts banknotes into personal clothing or a personal bag. Cash currently being counted is normal.',
        'limitation': 'Нужна видимая связь денег с предыдущей операцией; открытый ящик сам по себе недостаточен.',
    },
    'cash_not_counted': {
        'title': 'Передача наличных без видимого пересчёта',
        'section': 'Пункты 19(2), 31(4), 32',
        'rule': 'Приём и выдача требуют пересчёта; для неповреждённых упакованных пачек предусмотрены исключения.',
        'criterion': 'The entire handover of LOOSE banknotes is visible and the cashier directly stores or hands them over without either manual counting or using a counting machine. Do not flag sealed bundles, incomplete clips or counting outside sampled frames.',
        'limitation': 'Проверяется весь эпизод. Отсутствие пересчёта между редкими кадрами не доказывает нарушение.',
        'full_sequence': True,
    },
    'count_without_customer': {
        'title': 'Пересчёт без присутствия клиента',
        'section': 'Пункты 20, 31(4)',
        'rule': 'Пересчёт при приёме и перед выдачей выполняется в присутствии клиента.',
        'criterion': 'A customer hands cash to the cashier, visibly leaves, then the cashier starts counting those same banknotes. Do not flag internal cash preparation before a customer arrives.',
        'limitation': 'Внутренняя подготовка денег без клиента не считается этим событием.',
    },
    'damaged_collection_bag': {
        'title': 'Приём повреждённой инкассаторской сумки',
        'section': 'Пункты 24–25',
        'rule': 'При приёме проверяются сумка и пломбирование; повреждение требует вскрытия и пересчёта при инкассаторе.',
        'criterion': 'A visibly torn or already-open cash collection bag is accepted and put away while the collector leaves, without a visible opening and recount in their presence. A normal handbag is not a collection bag. Do not infer small seal damage from low resolution.',
        'limitation': 'Пломба и повреждение должны быть различимы, а последовательность приёма видна полностью.',
        'full_sequence': True,
    },
    'damaged_cash_package': {
        'title': 'Передача повреждённой упаковки инкассаторам',
        'section': 'Пункт 112',
        'rule': 'Инкассаторы не принимают пачки и мешки с обнаруженными дефектами упаковки.',
        'criterion': 'A collection team accepts visibly torn, open or defective packaged banknotes for transportation. The package defect and completed acceptance must both be clearly visible.',
        'limitation': 'Разрешённое вскрытие для пересчёта не считается дефектной передачей.',
    },
}

OBSERVATIONS = {
    'camera_phone': 'Модель отметила у кассира устройство, похожее на смартфон с камерой, внутри рабочей кассовой зоны.',
    'personal_belongings': 'Модель отметила предмет, похожий на личную сумку, рюкзак или кошелёк, на стороне кассира.',
    'cashier_left': 'По последовательности кадров модель предполагает уход кассира до завершения обслуживания клиента.',
    'cash_storage': 'Модель отметила возможное оставление денег завершённой операции вне предназначенного места хранения либо помещение денег в личные вещи.',
    'cash_not_counted': 'Модель отметила передачу разрозненных банкнот без различимого ручного или машинного пересчёта в показанной последовательности.',
    'count_without_customer': 'Модель отметила начало пересчёта переданных клиентом денег после его ухода.',
    'damaged_collection_bag': 'Модель отметила приём инкассаторской сумки с предполагаемым видимым дефектом без различимой необходимой проверки при инкассаторе.',
    'damaged_cash_package': 'Модель отметила передачу инкассаторам упаковки банкнот с предполагаемым видимым дефектом.',
}


def public_rules():
    return [dict(id=key, **{k: v for k, v in value.items() if k not in ('criterion', 'full_sequence')}, source=SOURCE) for key, value in RULES.items()]
