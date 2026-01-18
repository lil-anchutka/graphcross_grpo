# Graph Cross — среда и эксперименты с RL

В данном репозитории представлена синтетическая логическая среда **Graph Cross** и эксперименты по обучению LLM-агента с использованием reinforcement learning с верифицируемым бинарным вознаграждением.

Работа выполнена в формате исследовательского задания с фокусом на:
- дизайн среды,
- постановку RL-эксперимента,
- анализ динамики обучения и причин отсутствия прогресса при разреженном reward.

---

## Структура репозитория

```
task_graphcross/
├── src/
│   ├── base/
│   │   ├── env.py             
│   │   ├── data.py          
│   │   ├── verifier.py         
│   │   └── __init__.py
│   │
│   └── graphcross/
│       ├── __init__.py
│       ├── graphcross.py       # Реализация среды GraphCross:
│       │                      # генерация задач, логика 
│       ├── datasets.py         # Dataset-классы (train / eval) для GraphCross
│       ├── graphcross_prompt.py# шаблон формата задачи
│       └── graphcross_verifier.py
│                              # Проверка корректности решения (reward / correctness)
│
├── data/
│   └── eval/                   # Eval-наборы (jsonl с задачами)
│
├── notebooks/
│   └── experiment.ipynb        # Исследовательский ноутбук (запуски, отладка)
│
├── outputs_graphcross_stage1/
│   ├── checkpoint-400/         # Чекпоинт GRPO обучения (первый в двухэтапном обучении)
│   ├── checkpoint-600/
│   └── README.md               # Описание результатов stage1
│
├── graphcross_grpo_stage1_lora/
│   ├── adapter_model.safetensors # LoRA-адаптер (результат обучения)
│   ├── adapter_config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.json
│   ├── merges.txt
│   ├── added_tokens.json
│   ├── special_tokens_map.json
│   ├── chat_template.jinja
│   └── README.md               # Как использовать обученный адаптер
│
├── README.md                   # Общее описание проекта
├── REPORT.md                   # Главный аналитический отчет
└── .gitignore

```

---

## Отчет

Подробное описание задачи, среды, конфигураций обучения, логов и анализа результатов приведено в файле **REPORT.md**.

---

## Обучение модели

Код обучения и полные логи экспериментов представлены в Google Colab ноутбуке:

**https://drive.google.com/file/d/10a_xu-kM63V7H8R_BA2ThZVOeNO2uyhT/view?usp=sharing**

Ноутбук содержит:
- генерацию датасетов,
- запуск GRPO-тренера,
- логи обучения,
- промежуточные эксперименты с упрощением конфигурации.

---

## Примечание

В рамках экспериментов модель не продемонстрировала улучшения качества даже на упрощённых конфигурациях задач.  
Это рассматривается как осмысленный отрицательный результат и подробно анализируется в отчёте.

---

## Используемые технологии

- Python
- PyTorch
- HuggingFace Transformers
- Unsloth
- GRPO (Reinforcement Learning with Verifiable Rewards)
