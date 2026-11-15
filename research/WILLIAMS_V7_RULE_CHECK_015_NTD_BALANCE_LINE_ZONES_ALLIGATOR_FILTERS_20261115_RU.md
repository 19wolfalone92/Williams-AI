# RULE_CHECK_015 — NTD: Balance Line signals under Zone and Alligator filters

**Источник:** NTD, PDF SHA-256 `855eb052353c852f208436952b12e46d9a18badf7c1dac7ade7580d57b9f4bb4`, pages 103–114, Figures 8-7…8-23. Визуально сверены схемы signal shifting, buy/sell above/below Balance Line, и zone-filter summary.

## Базовая процедура (pp 103–110)
- Всегда читать справа налево.
- Для BUY анализировать highs; для SELL — lows.
- BUY base: текущий бар или последний бар с самым низким high. SELL base: текущий бар или бар с самым высоким low.
- От Balance Line: требуется один новый high для buy или один новый low для sell. К Balance Line: два соответствующих новых экстремума.
- Вариант «Special Blue Light» даёт перенос pending signal, когда новый бар действительно становится новой базой; обычный очередной high после утверждённой базы сам по себе не переносит buy stop. Аналогично для sell, через более высокий low как новую базу.
- Новый сигнал/перенос не равен ценовому триггеру и исполнению.

## Два дополнительных фильтра (pp 111, 114)
1. Не покупать ниже Alligator mouth; не продавать выше Alligator mouth. Иначе сигнал оказывается «в пасти»/против компаса Alligator.
2. Если ищем BUY в Red Zone, удваиваем число требуемых более высоких highs: вместо обычного одного/двух требуется два/четыре в зависимости от положения относительно Balance Line.
3. Если ищем SELL в Green Zone, удваиваем число требуемых более низких lows: вместо обычного двух/одного требуется четыре/два соответственно.
4. В Green Zone для BUY и Red Zone для SELL используются обычные требования; Gray Zone — обычные требования, если другие условия не запрещают сигнал.

## Важное разделение
- Базовый count определяется сначала положением price относительно Balance Line; затем Zone может удвоить count только в adverse-side сочетаниях: BUY in Red Zone / SELL in Green Zone.
- Alligator is a separate outer-mouth directional filter.
- Балансовая линия не тождественна Teeth, а Zone не тождественна AO/AC отдельному signal.

**Статус:** source spot-check; independent experts=0; no chart gold; no SFT approval.
