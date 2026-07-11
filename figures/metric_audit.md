# Figure and metric audit — v06.1

## Исправленный блокер

Старая Figure 3 и текст использовали разные метрики:

- текст §6.8 и `head_causal_atlas.csv`: `valid_tensor_rel_percent`;
- старая фигура: `last_token_rel_median * 100`.

Обе метрики были рассчитаны корректно, но описывали разные агрегаты и поэтому не могли сравниваться напрямую.

## Единая метрика v06.1

Figure 3a и текст §6.8 теперь используют:

```text
valid_tensor_rel_percent
```

Это относительная L2-пертурбация всего валидного residual-тензора в батче из 64 промптов.

Горизонтальная ось выровнена относительно первого residual после исходной головы:

```text
target_index - (source_layer + 1)
```

## Проверка трёх примеров

### QK L2H5

- first post-source residual: 3.5275876460%
- peak: 20.2874181747%
- peak target index: 22
- relative delay: 19 layers
- final logit effect: 2.1046278998%
- internal/final ratio: 9.6394323084x

### QK L17H2

- first post-source residual: 0.2184995739%
- peak: 5.5866054299%
- peak target index: 23
- relative delay: 5 layers
- final logit effect: 2.1344020963%
- downstream amplification: 25.5680381027x

### VO L16H10

- first post-source residual: 0.3776880279%
- peak: 5.7826799058%
- peak target index: 23
- relative delay: 6 layers
- final logit effect: 2.1355949342%
- downstream amplification: 15.3107312880x

Скрипт автоматически сравнивает peak values/indices из `propagation_profiles.csv` и `head_causal_atlas.csv` и завершает работу с ошибкой при несогласованности.

## Figure 3b

Правая панель теперь показывает не абсолютный layer index, а per-head delay до максимальной valid-tensor perturbation:

- QK median delay: 10 layers
- VO median delay: 9 layers

Это напрямую соответствует таблице §6.8.

## Косметические правки

- убраны `Figure N` и общие заголовки внутри изображений;
- сохранены только panel titles `(a)/(b)`;
- убрана надпись, наезжавшая на правый блок Figure 1;
- PDF и PNG создаются одним скриптом;
- PDF-файлы повторно отрендерены при 180 DPI и визуально проверены;
- клиппинг, наложение текста и сломанные глифы не обнаружены.

## Итоговые файлы

- `generate_matrix_program_figures_v06_1_fixed.py`
- `figures_v06_1_fixed/*.pdf`
- `figures_v06_1_fixed/*.png`
- `figures_v06_1_fixed/figure_metric_audit.json`
- `статья_матричная_программа_v06_1_fixed_ru.md`
