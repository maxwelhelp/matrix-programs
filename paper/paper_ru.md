# Executable Matrix Programs: faithful weight-derived replacement and factorized intervention in pretrained transformers

**Русский arXiv-черновик v0.7**

**Автор:** Maxim Vladimirovich Zhivotok  
**Аффилиация:** Independent Researcher, Ukraine  
**Email:** maxwelhelp@gmail.com  
**Код и результаты:** https://github.com/maxwelhelp/matrix-programs

---

## Аннотация

Мы представляем метод извлечения **исполняемых компонентных матричных программ** из предобученного decoder-only трансформера без переобучения и без изменения его весов. Для RoPE-внимания метод строит weight-derived QK-routing таргет \(M^{\mathrm{aug}}_{qk}[d]\), зависящий от относительного расстояния, и VO-payload карту \(C^{\mathrm{aug}}_{vo}\); для SwiGLU-MLP он строит точную атомную программу с gate/read/write факторизацией. В отличие от input-specific локально-линейных представлений, эти структурные объекты извлекаются непосредственно из весов и переиспользуются между входами. На Qwen2.5-0.5B-Instruct мы одновременно заменяем программным вычислением все 336 attention-голов и все 24 MLP-подслоя внутри native forward: на 64 промптах медианная относительная ошибка полного вектора logits составляет **0.258%**, p95 — **0.566%**, максимум — **0.923%**, при этом top-1 сохраняется в **100%** случаев. На основе того же интерфейса мы строим полный factorized causal atlas: для всех 336 голов отдельно измеряются exact replacement, QK-off и VO-off, а для всех 24 MLP — exact replacement и MLP-off. Медианные отношения intervention effect к observed replacement discrepancy составляют **24.89×** для QK, **20.62×** для VO и **152.05×** для MLP. Downstream propagation profiles показывают, что у **75.0%** QK-интервенций и **71.4%** VO-интервенций пиковый внутренний эффект достигается как минимум через четыре слоя после источника, а медианная задержка до пика равна **10** и **9** слоям соответственно. Тем самым matrix program выступает не только как аналитическая запись компонента, но и как **исполняемый интерфейс** для faithful replacement, factorized intervention и анализа распределённого вычисления по глубине модели.

Во всей статье используется термин **replacement discrepancy**: native identity control в текущей реализации является sanity check, но не независимой оценкой численного нижнего порога. Per-experiment сводки по промптам используют сохранённую в коде агрегацию `torch.median`, которая при чётном числе наблюдений возвращает нижний из двух центральных элементов. Медианы по строкам опубликованных head- и MLP-atlas таблиц являются обычными статистическими медианами и при чётном числе компонентов усредняют два центральных значения.

---

## 1. Введение

Механистическая интерпретируемость стремится восстановить вычислительную структуру обученной сети: понять, какие компоненты модели реализуют конкретные преобразования и как эти преобразования совместно формируют ответ. Для трансформеров уже существуют важные линии работ: QK/OV circuit analysis, activation patching и causal tracing, sparse autoencoders, input-specific locally linear mappings, singular-vector decomposition внутренних компонентов и RASP-style decompilation. Однако остаётся практический вопрос: можно ли извлечь из весов современной языковой модели **явную и исполняемую** компонентную программу, заново выполнить её внутри исходного forward и заменить ею не один выбранный тензор, а весь attention+MLP backbone?

Современный decoder transformer усложняет задачу деталями реализации: pre-normalization, RMSNorm, RoPE, GQA, q/k/v biases, causal masking, gated SwiGLU MLP и различия dtype. Каждая часть по отдельности известна, но faithful extractor должен правильно совместить их с native implementation. В этой работе мы строим такой extractor для Qwen2.5-style архитектуры и проверяем не только локальную реконструкцию, но и **глобальную исполняемую замену** внутри настоящего inference.

### 1.1 Основные вклады

1. **Единое исполняемое weight-derived представление программ компонентов.** Опираясь на уже известные QK/OV circuit decomposition и относительную RoPE-формулировку (Elhage et al., 2021; Su et al., 2021), мы реализуем переиспользуемую программу для современного стека RoPE/GQA/RMSNorm/bias/SwiGLU: индексированные относительным расстоянием аффинно-билинейные QK-таргеты \(M^{\mathrm{aug}}_{qk}[d]\), аффинную VO-карту \(C^{\mathrm{aug}}_{vo}\) и точную атомную программу SwiGLU, сохраняющую native-нелинейность во входозависимых коэффициентах над статическими rank-1 атомами с факторизацией gate/read/write. Вкладом является интегрированное исполняемое представление и его faithful reinsertion, а не новизна самих QK/OV decomposition, RoPE identity или алгебраического раскрытия SwiGLU. В отличие от input-specific Jacobian mappings (Golden, 2025), структурные объекты выводятся из весов один раз и переиспользуются между входами; в отличие от transcoders (Dunefsky et al., 2024), surrogate-модель не обучается.
2. **Глобальная исполняемая замена.** Все 336 attention-голов и все 24 MLP-подслоя Qwen2.5-0.5B-Instruct заменяются weight-derived программным вычислением внутри native forward; полный backbone сохраняет top-1 на всех 64 промптах при медианной ошибке logits **0.258%**.
3. **Полный factorized causal atlas attention.** Для каждой головы отдельно измеряются exact replacement, QK-off, VO-off и downstream propagation, что разделяет routing и payload.
4. **Полный atlas MLP.** Для всех 24 MLP измеряются exact replacement и causal removal; интервенции оказываются на два–три порядка сильнее replacement discrepancy.
5. **Калибровка effect-to-replacement-discrepancy.** Каждый intervention effect нормируется на discrepancy того же executable replacement interface: медианные отношения составляют **24.89×** для QK, **20.62×** для VO и **152.05×** для MLP. Мы используем это как переиспользуемую практику для отделения эффектов вмешательства от артефактов реализации замены, а не как самостоятельную теорему причинной идентификации.
6. **Downstream profiling.** Мы измеряем не только финальный logits effect, но и изменение residual stream после каждого последующего слоя, выявляя отложенные и подготовительные компоненты. Медианная задержка до пика равна 10 слоям для QK-интервенций и 9 слоям для VO-интервенций.
7. **Верифицированный extractor современного стека.** Реализация учитывает RoPE, RMSNorm, GQA, q/k/v biases и SwiGLU и публикует полные машинно-читаемые отчёты. Прямой контроль показывает эмпирическую необходимость сворачивания обученных biases: медианная ошибка QK-формы без bias равна **105%**, тогда как полная аффинная форма даёт **0.0526%**.

Отдельно мы описываем **предварительную gate lens**. Поскольку активация SwiGLU-нейрона модулируется \(\operatorname{silu}(W_g x)\), мы используем gate-проекцию \(W_g\) как качественный probe селективности нейрона, а не рассматриваем up-проекцию \(W_u\) как единственный селектор. Линза используется только в иллюстративном примере раздела 6.5 и не позиционируется как подтверждённый основной вклад до проведения количественного benchmark (ограничение 9).

### 1.2 Что мы не заявляем

Мы не заявляем новизну общей идеи QK/OV decomposition, алгебраического раскрытия SwiGLU, homogeneous coordinates как математического приёма, RoPE relative-position identity, activation ablation вообще, представления трансформера как матричной системы вообще, разрешения суперпозиции или ускорения inference. Мы также не заявляем универсальную поддержку любых архитектур без architecture adapter.

---

## 2. Архитектура и обозначения

Рассмотрим decoder-only трансформер с residual dimension \(H\). Состояние позиции \(t\) перед компонентом слоя обозначим \(x_t\in\mathbb{R}^{H}\).

### 2.1 RMSNorm

\[
\widetilde{x} = \operatorname{RMSNorm}_{\gamma}(x) = \frac{x}{\sqrt{\operatorname{mean}(x^2)+\varepsilon}}\odot\gamma.
\]

### 2.2 Однородные координаты

Для аффинной проекции \(z=Wx+b\) вводим

\[
x_{\mathrm{aug}}=\begin{bmatrix}x\\1\end{bmatrix},\qquad W_{\mathrm{aug}}=\begin{bmatrix}W & b\end{bmatrix},
\]

так что \(z=W_{\mathrm{aug}}x_{\mathrm{aug}}\). В companion reconstruction run QK-форма со свёрнутыми biases имела медианную ошибку **0.0526%**, тогда как контроль без biases — **105%**. Для Qwen2.5 homogeneous coordinates являются необходимой частью faithful circuit target.

### 2.3 RoPE-конвенция

В реализации тензоры хранятся строками, а RoPE применяется как \(vR_p^\top\). Формулы ниже записаны в эквивалентной столбцовой конвенции. Корректность порядка матриц проверяется direct reconstruction against native forward.

---

## 3. Weight-derived component programs

### 3.1 QK: routing program

Для attention-головы размерности \(D\):

\[
q_i^{\mathrm{pre}} = W_q\widetilde{x}_i+b_q,\qquad k_j^{\mathrm{pre}} = W_k\widetilde{x}_j+b_k.
\]

После RoPE:

\[
q_i=R_iq_i^{\mathrm{pre}},\qquad k_j=R_jk_j^{\mathrm{pre}}.
\]

Pre-softmax score:

\[
s_{ij}=\frac{q_i^\top k_j}{\sqrt{D}} = x_{\mathrm{aug},i}^{\top} M^{\mathrm{aug}}_{qk}[i-j] x_{\mathrm{aug},j},
\]

где

\[
M^{\mathrm{aug}}_{qk}[d] = \frac{(W_q^{\mathrm{aug}})^\top R_{\mathrm{rel}}[d] W_k^{\mathrm{aug}}}{\sqrt{D}}.
\]

Runtime-path может исполнять эту же программу в факторизованной форме через augmented q и k, без обязательной материализации полной \((H+1)\times(H+1)\) матрицы для каждой дистанции.

### 3.2 VO: payload program

\[
C^{\mathrm{aug}}_{vo}=W_oW_v^{\mathrm{aug}},\qquad \operatorname{payload}_j=C^{\mathrm{aug}}_{vo}x_{\mathrm{aug},j},
\]

\[
y_i=\sum_j A_{ij}\operatorname{payload}_j.
\]

QK определяет routing, а VO — переносимое содержимое и направление записи.

### 3.3 SwiGLU MLP

\[
y=W_d\left[\operatorname{silu}(W_g\widetilde{x})\odot (W_u\widetilde{x})\right].
\]

Покомпонентно:

\[
y=\sum_{j=1}^{m} a_j(x)W_d[:,j],\qquad a_j(x)=\operatorname{silu}(W_{g,j}\widetilde{x})\cdot (W_{u,j}\widetilde{x}).
\]

Мы разделяем нейрон на gate, read и write:

\[
\operatorname{gate}_j(x)=\operatorname{silu}(W_{g,j}\widetilde{x}),\quad
\operatorname{read}_j(x)=W_{u,j}\widetilde{x},\quad
\operatorname{write}_j=W_d[:,j],
\]

и определяем статический rank-1 atom

\[
B_j = W_d[:,j]\otimes W_u[j,:].
\]

---

## 4. Исполняемая замена и факторизованные вмешательства

Для выбранного множества attention-голов \(S_L\) слоя \(L\) вычисляются native-вклад \(Y^{\mathrm{native}}_{S_L}\) и program-вклад \(Y^{\mathrm{program}}_{S_L}\). Выход attention-модуля заменяется на

\[
Y^{\mathrm{patched}} = Y^{\mathrm{module}} - Y^{\mathrm{native}}_{S_L} + Y^{\mathrm{program}}_{S_L}.
\]

Для MLP native output заменяется на

\[
Y^{\mathrm{program}}_{\mathrm{MLP}} = \sum_j a_j(x)W_d[:,j].
\]

Глобальный experiment одновременно устанавливает hooks на все 24 attention-подслоя и все 24 MLP-подслоя, то есть заменяет весь attention+MLP backbone. Layer norms, embeddings, residual additions и LM head остаются native.

Вмешательства определяются следующим образом:

- **QK-off:** content-dependent QK scores выбранной головы зануляются до softmax.
- **VO-off:** payload выбранной головы зануляется при сохранённом routing.
- **MLP-off:** output выбранного MLP зануляется.

Основная метрика exact replacement:

\[
E_{\mathrm{logit}} = 100\cdot \frac{\|\ell_{\mathrm{patched}}-\ell_{\mathrm{base}}\|_2}{\|\ell_{\mathrm{base}}\|_2}.
\]

Также измеряются KL, JS divergence, cosine similarity logits, top-1 preservation и top-5 overlap.

Мы используем термин **replacement discrepancy**. Native identity control в текущей реализации является sanity check и не используется как независимая оценка численного нижнего порога.

---

## 5. Экспериментальная установка

### 5.1 Модель

`Qwen/Qwen2.5-0.5B-Instruct`:

- 24 слоя;
- 14 attention heads на слой (всего 336);
- 2 KV-heads;
- residual dimension \(H=896\);
- head dimension \(D=64\);
- SwiGLU intermediate dimension \(m=4864\);
- fp16;
- eager attention.

### 5.2 Prompt suite

64 коротких next-token prompts:

- Capitals: 12
- Factual: 12
- Math: 10
- Code: 10
- Continuation: 10
- Russian: 4
- Ukrainian: 4
- German: 1
- Spanish: 1

### 5.3 Масштаб экспериментов

- 1662 основных experiments;
- 106,368 experiment–prompt evaluations;
- глобальные замены;
- полный 336-head atlas;
- полный 24-MLP atlas;
- 24 layer-group profiles;
- downstream propagation profiles.

### 5.4 Воспроизводимость

Для arXiv-версии в репозитории должны быть зафиксированы:

- `torch` и `transformers` версии;
- ревизия модели `Qwen/Qwen2.5-0.5B-Instruct`;
- seed;
- железо: **Tesla P40 24GB**;
- время полного прогона;
- команды запуска;
- SHA256 основного скрипта;
- исправленный atom-split script с независимыми random seeds.

---

## 6. Результаты

### 6.1 Companion reconstruction

| Компонент | Median relative error |
|---|---:|
| QK scores с RoPE+bias | 0.0526% |
| Attention probabilities | 0.0599% |
| VO head output | 0.0277% |
| MLP output | 0.0426% |
| QK без bias folding | 105% |

No-bias контроль показывает, что для Qwen2.5 сворачивание biases через homogeneous coordinates является необходимым, а не косметическим приёмом.

### 6.2 Глобальная исполняемая замена

| Замена | N | Median | p95 | Max | Median KL | Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| Все attention | 64 | 0.254% | 0.477% | 0.885% | 3.38e-5 | 100% |
| Все MLP | 64 | 0.241% | 0.419% | 0.883% | 2.68e-5 | 100% |
| Attention + MLP | 64 | **0.258%** | **0.566%** | **0.923%** | **3.09e-5** | **100%** |

Для полного backbone медианная cosine similarity logits равна **0.999990**, а медианный top-5 overlap — **1.0**.

![Схема метода](../figures/fig1_method_overview.png)

*Рисунок 1. Извлечение weight-derived программы, её исполняемая замена внутри native forward и факторизованные вмешательства.*

![Глобальная замена по типам промптов](../figures/fig4_global_replacement_by_suite.png)

*Рисунок 4. Ошибка глобальной замены attention+MLP backbone по категориям промптов; показаны медиана и максимум внутри каждой категории.*

### 6.3 Одиночная исполняемая замена по всей модели

| Объект | Число компонентов | Median replacement discrepancy | p95 | Max | Aggregate top-1 preservation |
|---|---:|---:|---:|---:|---:|
| Attention head | 336 | 0.137% | 0.201% | 0.265% | 100% |
| MLP layer | 24 | 0.139% | 0.192% | 0.243% | 99.935% |
| Full transformer layer | 24 | 0.139% | 0.214% | 0.241% | >99.9% |

### 6.4 Полный head atlas

| Метрика | QK-off | VO-off |
|---|---:|---:|
| Median logit effect | 3.280% | 2.883% |
| p95 | 6.859% | 6.664% |
| Max | 23.649% | 11.403% |
| Median effect/replacement ratio | 24.89× | 20.62× |
| Heads with ratio >10× | 92.86% | 84.23% |
| Aggregate top-1 change | 6.12% | 5.82% |
| Heads changing top-1 at least once | 96.13% | 92.56% |

![Полный атлас голов](../figures/fig2_head_heatmaps.png)

*Рисунок 2. Полный атлас 24×14. Слева — observed replacement discrepancy каждой головы; справа — QK-off effect той же головы.*

### 6.5 Качественная иллюстрация: `"The capital of France is"`

**Эта секция иллюстративная и не входит в основные claims. Gate-lens остаётся exploratory инструментом (см. Limitations).** Тем не менее для mech-interp статьи полезно показать, что исполняемый интерфейс делает вычисление читаемым на конкретном примере.

Промпт `"The capital of France is"` даёт предсказание `" Paris"`. На late layers извлечённая программа показывает три уровня структуры.

**Attention — какие головы вносят наибольший вклад и откуда читают.**

- **L21H6:** вклад `+1.094`; источники: `'France' (A=0.73)`, `'The' (0.15)`, `'is' (0.07)`.
- **L21H1:** вклад `+0.279`; источники: `'France' (0.41)`, `'The' (0.36)`.
- **L20H5:** вклад `+0.201`; источники: `'France' (0.52)`, `'The' (0.38)`, `'capital' (0.05)`.

Это делает явным не только routing (куда смотрит голова), но и payload, поскольку переносимое содержимое задаётся картой \(C_{vo}^{aug}x_{\mathrm{aug},j}\).

**MLP atoms — какие late-layer atoms активны и какую тематику несут.**

- **L18 n3178:** `a=-18.25`; gate-lens: `towns, hom`.
- **L20 n4520:** `a=-30.43`; gate-lens: `city, .city, _city`.
- **L20 n3433:** `a=-9.28`; читает: `congressional, 白宫, 政府`; пишет: `Washington, DC, 华盛顿`.
- **L21 n4090:** `a=+30.97`; читает: `法国, French, France`; пишет: `法国, French, France`.
- **L22 n2750:** `a=-6.73`; читает: `French, Prix, 法国`; пишет: `巴黎, Jean, Paris`.
- **L23 n1465:** `a=+47.10`; читает: `French, France, 法国, Paris`.

Иллюстративно это образует содержательную цепочку `towns → city → France → Paris`.

**Причинная верификация из предыдущего exploratory прогона.**

- **L23 n1465:** `Δlogit(real) = -8.406`; `real/random = 76.9`; `real/unrelated = 7.0`.
- **L21 n4090:** `Δlogit(real) = +0.719`; `real/random = 3.5`; `real/unrelated = 3.1`.

Так как текущий held-out atom suite в архиве пережил seed-bug в random controls, этот пример следует воспринимать как qualitative illustration, а не как часть центрального количественного результата.

### 6.6 Полный MLP atlas

| Метрика | Значение |
|---|---:|
| Median replacement discrepancy | 0.139% |
| Median MLP-off effect | 23.44% |
| p95 MLP-off | 81.97% |
| Max MLP-off | 99.51% |
| Median effect/replacement ratio | 152.05× |
| Aggregate top-1 change | 38.15% |


### 6.7 Layer-group interventions

| Интервенция | Median logit effect | Median top-1 change rate |
|---|---:|---:|
| QK всех голов слоя off | 13.91% | 17.19% |
| VO всех голов слоя off | 10.05% | 18.75% |
| MLP off | 23.44% | 33.59% |
| VO + MLP off | 27.44% | 36.72% |

### 6.8 Downstream propagation и подготовительные компоненты

| Статистика | QK | VO |
|---|---:|---:|
| Median delay до пика | 10 | 9 |
| Peak delayed ≥4 layers | 75.0% | 71.4% |
| Median peak/first amplification | 17.98× | 17.19× |
| Amplification >10× | 76.2% | 73.5% |

В этой секции и на рисунке 3 используется одна и та же метрика: `valid_tensor_rel_percent`, то есть относительная L2-пертурбация всего валидного residual-тензора в батче из 64 промптов. Горизонтальная ось рисунка 3a выровнена относительно первого residual state после исходной головы: `target_index − (source_layer + 1)`.

Подготовительные примеры:

- **QK L2H5:** final logits `2.10%`; первый post-source residual `3.53%`; peak residual `20.29%` на target residual index `22`, то есть через `19` слоёв после первого post-source residual; internal/final ratio `9.64×`.
- **QK L17H2:** final logits `2.13%`; первый post-source residual `0.219%`; peak residual `5.59%` на target residual index `23`, то есть с задержкой `5` слоёв; amplification `25.57×`.
- **VO L16H10:** final logits `2.14%`; первый post-source residual `0.378%`; peak residual `5.78%` на target residual index `23`, то есть с задержкой `6` слоёв; amplification `15.31×`.

![Downstream propagation](../figures/fig3_propagation_profiles.png)

*Рисунок 3. (a) Выбранные propagation trajectories, рассчитанные по `valid_tensor_rel_percent` и выровненные относительно первого post-source residual. (b) Распределение per-head задержки до максимальной valid-tensor perturbation; медианы равны 10 слоям для QK и 9 слоям для VO.*

Эти результаты поддерживают тезис, что слабый прямой final-logit effect не означает малую вычислительную роль компонента: многие головы готовят состояние для последующих слоёв.

---

## 7. Обсуждение

### 7.1 Что является главным объектом работы

Главный объект работы — не матрица сама по себе, а связка

\[
\text{weight-derived structure} + \text{input-dependent execution} + \text{executable replacement} + \text{factorized intervention}.
\]

Статические структуры: \(M_{qk}^{aug}[d]\), \(C_{vo}^{aug}\), \(B_j\). Динамические величины: hidden states, attention probabilities и MLP coefficients.

### 7.2 Почему global replacement важнее локального тождества

Алгебраическая формула может быть верной, но реализация может ошибиться в RoPE orientation, norm placement, head layout, GQA mapping, masking или dtype. Global replacement одновременно тестирует всю цепочку.

### 7.3 Сравнение с activation patching

Activation patching обычно использует cached clean/corrupted activations. Здесь значение компонента **пересчитывается** из текущего hidden state через weight-derived program. Это делает возможными faithful replacement, QK-only intervention, VO-only intervention и MLP intervention в одном интерфейсе. Мы не утверждаем, что activation patching принципиально неспособен перехватывать внутренние Q/K/V tensors; отличие состоит в том, что наша замена задаётся точной program representation, а не cached counterfactual activation.

### 7.4 Подготовительные компоненты и distributed computation

Отложенные пики downstream effect согласуются с идеей межслойной коммуникации и подготовительных компонентов. Однако propagation metric измеряет perturbation в downstream graph в целом и не выделяет конкретные causal edges. Следующим шагом может быть объединение program interface с path patching, path expansion или subspace tracing.

### 7.5 Суперпозиция

Rank-1 MLP atoms являются native вычислительными объектами, но не обязательно monosemantic features. Работа не разрешает суперпозицию; SAE или transcoder можно применять поверх program representation как дополнительный слой анализа.

---

## 8. Related work

### 8.1 Transformer circuits

Работы по transformer circuits задали язык QK/OV decomposition, read/write maps и анализа attention-голов. Мы продолжаем эту линию, но переносим её в faithful extractor для RoPE/RMSNorm/GQA/bias/SwiGLU и проверяем не только analysis, но и global executable replacement.

### 8.2 Locally linear mappings

Detached-Jacobian подходы, в частности Golden, строят почти точные input-specific линейные карты модели или её компонентов. Наш объект отличается: structural targets выводятся из весов и переиспользуются между входами, а input-dependent execution остаётся явной в attention probabilities и MLP coefficients.

### 8.3 Singular-vector / component decomposition

Beyond Components исследует singular directions и подфункции внутри attention и MLP. Наша factorization не конкурирует с SVD как поиском подпространств: она задаёт native executable representation, поверх которой SVD может применяться дополнительно.

### 8.4 Activation patching и causal tracing

Activation patching и causal tracing локализуют причинно значимые компоненты, подменяя активации из других запусков или искусственными значениями. Мы используем тот же interventional дух, но заменяем компонент не cached activation, а значением, пересчитанным через weight-derived программу на текущем hidden state.

### 8.5 Causal scrubbing

Causal scrubbing проверяет интерпретационные гипотезы, заменяя промежуточные значения ресэмплированными величинами, согласованными с гипотезой. Наша работа близка по духу к mechanism replacement, но объект замены другой: вместо hypothesis-driven resampling мы используем точную weight-derived программу компонента. Это делает цель более узкой, но и более строгой: мы проверяем faithful replacement и factorized intervention для native component computation.

### 8.6 Sparse autoencoders и transcoders

SAE строят обучаемый feature basis поверх активаций. Transcoders обучают sparse surrogate для компонента, чаще всего MLP, чтобы аппроксимировать его input-output mapping и облегчить circuit analysis. Наш подход отличается тем, что не обучает surrogate и не делает шагов оптимизации: программа извлекается напрямую из весов и уже даёт медианную replacement discrepancy **0.139%** для всех 24 MLP без обучения.

### 8.7 Talking Heads и inter-layer communication

Работы по inter-layer communication анализируют низкоранговые каналы передачи информации между слоями и головами. Наш full atlas поддерживает сам факт широко распространённых delayed downstream effects, но не выделяет конкретные communication subspaces.

### 8.8 RASP decompilation и white-box модели

RASP decompilation извлекает символические программы из небольших алгоритмических трансформеров. White-box architectures строят новые интерпретируемые модели. Мы работаем с предобученной языковой моделью и строим матричную, а не символическую программу её native-компонентов.

---

## 9. Ограничения

1. Проверена одна модель — Qwen2.5-0.5B-Instruct.
2. Prompt suite содержит 64 коротких next-token prompts, а не большой benchmark.
3. Нет измерения perplexity/loss на корпусе.
4. Нет long-context проверки больших RoPE distances.
5. Program path использует float32 для части операций, тогда как native модель работает в fp16.
6. Native identity control в attention является sanity check, а не независимой оценкой численного нижнего порога.
7. Propagation показывает downstream perturbation, но не отдельные causal edges.
8. Held-out atom experiment не входит в основные claims: в исходном run три random controls дублировались из-за seed-bug; исправленная версия кода публикуется отдельно.
9. Gate-lens остаётся exploratory и используется только для qualitative illustration.
10. Нет claim об inference speedup.

---

## 10. Заключение

Мы представили исполняемое component-level representation предобученного Qwen2.5 transformer. Все attention и MLP computations модели были заменены программными вычислениями из весов с медианной ошибкой logits **0.258%** и **100%** сохранением top-1 на 64 промптах. Full head atlas показал, что QK и VO interventions оказывают effects в десятки раз выше replacement discrepancy, а MLP interventions — более чем в сто раз выше. Downstream profiles выявили, что большинство head effects достигает пика через много слоёв после источника. Следовательно, component matrix program является не только способом переписать формулу, но и практическим исполняемым интерфейсом для faithful replacement, factorized intervention и анализа распределённого вычисления по глубине модели.

---

## Приложение A. Генерация фигур

Фигуры рисуются автоматически из CSV результатов скриптом:

```bash
python generate_matrix_program_figures_v06_1_fixed.py \
  --results-dir outputs/qwen_matrix_program_final_article_v4/final_article \
  --out-dir figures_v06_1_fixed
```

Скрипт создаёт векторные PDF и PNG-файлы:

- `fig1_method_overview.pdf`
- `fig2_head_heatmaps.pdf`
- `fig3_propagation_profiles.pdf`
- `fig4_global_replacement_by_suite.pdf`

## Приложение B. Воспроизводимость

Минимальный набор файлов для GitHub:

- основной extractor/patching script;
- исправленный atom-split script с независимыми random seeds;
- `README.md` с точной командой запуска;
- `all_experiment_summaries.csv`;
- `all_prompt_metrics.csv`;
- `head_causal_atlas.csv`;
- `mlp_causal_atlas.csv`;
- `layer_group_atlas.csv`;
- `propagation_profiles.csv`;
- `experiment_manifest.json`;
- `final_prompts.json`;
- версии окружения и hardware note.

---

## Библиография

[1] Elhage, N., Nanda, N., Olsson, C., et al. *A Mathematical Framework for Transformer Circuits*. Transformer Circuits Thread, 2021.

[2] Golden, J. R. *Equivalent Linear Mappings of Large Language Models*. arXiv:2505.24293, 2025.

[3] Ahmad, A., Joshi, A., Modi, A. *Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits*. arXiv:2511.20273, 2025.

[4] Heimersheim, S., Nanda, N. *How to Use and Interpret Activation Patching*. arXiv:2404.15255, 2024.

[5] Zhang, F., Nanda, N. *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods*. arXiv:2309.16042, 2023.

[6] Chan, L., Garriga-Alonso, A., Goldowsky-Dill, N., et al. *Causal Scrubbing: a method for rigorously testing interpretability hypotheses*. AI Alignment Forum / Redwood Research, 2022.

[7] Geiger, A., Ibeling, D., Zur, A., et al. *Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability*. arXiv:2301.04709, 2023/2025.

[8] Bricken, T., Templeton, A., Batson, J., et al. *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning*. Transformer Circuits Thread, 2023.

[9] Dunefsky, J., Chlenski, P., Nanda, N. *Transcoders Find Interpretable LLM Feature Circuits*. arXiv:2406.11944, 2024.

[10] Merullo, J., Eickhoff, C., Pavlick, E. *Talking Heads: Understanding Inter-layer Communication in Transformer Language Models*. arXiv:2406.09519, 2024.

[11] Huang, X., Bakalova, A., Bhattamishra, S., Merrill, W., Hahn, M. *Discovering Interpretable Algorithms by Decompiling Transformers to RASP*. arXiv:2602.08857, 2026.

[12] Yu, Y., Buchanan, S., Pai, D., et al. *White-Box Transformers via Sparse Rate Reduction*. arXiv:2306.01129, 2023.

[13] Su, J., Lu, Y., Pan, S., et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv:2104.09864, 2021.

[14] Shazeer, N. *GLU Variants Improve Transformer*. arXiv:2002.05202, 2020.

[15] Qwen Team. *Qwen2.5 Technical Report*. arXiv:2412.15115, 2024.
