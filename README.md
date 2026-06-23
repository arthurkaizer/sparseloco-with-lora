# SparseLoCo + LoRA — Fine-tuning no MedQA (USMLE)

Projeto Final do Seminário de Deep Learning.

Aplicação do otimizador **SparseLoCo** — proposto no contexto de pré-treinamento distribuído de LLMs e utilizado no treinamento do Covenant-72B — ao cenário de **fine-tuning com LoRA** no benchmark médico MedQA (USMLE).

---

## Pergunta de Pesquisa

> O SparseLoCo+LoRA atinge accuracy comparável ao LoRA centralizado no MedQA? Se sim, o SparseLoCo viabiliza fine-tuning distribuído sem degradar a qualidade do modelo.

---

## Artigos de Base

- **SparseLoCo** — Sarfi et al. (2025). *Communication Efficient LLM Pre-Training with SparseLoCo.* arXiv:2508.15706
- **Covenant-72B** — Lidin et al. (2026). *Pre-Training a 72B LLM with Trustless Peers Over-the-Internet.*

---

## Abordagem

O SparseLoCo foi proposto e validado exclusivamente para pré-treinamento com gradientes densos. Este trabalho explora uma aplicação nova: fine-tuning com LoRA em domínio específico (médico), avaliando se a compressão Top-k com error feedback mantém qualidade comparável ao fine-tuning centralizado.

Comparamos duas abordagens no benchmark MedQA (USMLE):

| Abordagem | Parâmetros treináveis | Comunicação |
|---|---|---|
| AdamW + LoRA (baseline) | LoRA adapters | centralizado, sem compressão |
| SparseLoCo + LoRA | LoRA adapters | Top-k 10% comprimido, distribuído simulado |

**Modelo principal:** Llama-3.1-8B com QLoRA 4-bit (T4/A100, requer login HuggingFace)
**Modelo final:** Covenant-72B com QLoRA 4-bit (requer A100 80GB)
**Benchmark:** MedQA — questões USMLE (A/B/C/D/E, chance=20%)
**Workers simulados:** R=4, executados em série na mesma máquina

---

## Acesso ao Modelo (HuggingFace)

O Llama-3.1-8B é um modelo **gated** da Meta. Para acessá-lo:

1. Crie uma conta em `huggingface.co`
2. Acesse `huggingface.co/meta-llama/Llama-3.1-8B` e aceite a licença (aprovação automática)
3. Gere um token em `huggingface.co/settings/tokens` (tipo Read)
4. No notebook, cole o token na célula de autenticação (célula 0)

---

## Configuração do Experimento

| Parâmetro | Valor | Justificativa |
|---|---|---|
| `MODEL_ID` | Llama-3.1-8B | Mesma família LLaMA-3 do Covenant-72B |
| `B` | 4 | Llama-3.1-8B em 4-bit ocupa ~4GB |
| `MAX_LEN` | 512 | Cobre a maioria das questões MedQA |
| `LORA_R` | 16 | Capacidade suficiente para adaptação |
| `LR_INNER` | 2e-4 | Padrão para fine-tuning com LoRA |
| `H` | 30 | Inner steps por round |
| `T` | 20 | Outer rounds |
| `TOPK_K` | 0.10 | 90% de compressão nos pseudo-gradientes |
| `MAX_GRAD_NORM` | 1.0 | Gradient clipping |

---

## Estrutura

```
sparseloco-with-lora/
└── sparseloco-with-lora.ipynb    # notebook principal
```

---

## Como Executar

### Dependências

```python
!pip install torch transformers peft datasets matplotlib tqdm scikit-learn bitsandbytes>=0.46.1 accelerate
```

### Hardware

| Modelo | VRAM necessária |
|---|---|
| Llama-3.1-8B (4-bit) | T4 16GB ✅ |
| Covenant-72B (4-bit) | A100 80GB |
| TinyLlama-1.1B (fallback) | qualquer GPU |

### Executar

1. Abrir no Google Colab com **T4 GPU**
2. Montar Google Drive (feito automaticamente na célula de config)
3. Preencher token HuggingFace na célula 0
4. Rodar todas as células em ordem

Os checkpoints são salvos no Google Drive após cada round.

---

## Limitações

- Workers simulados em série na mesma máquina, sem comunicação real pela rede
- Llama-3.1-8B é um modelo intermediário — experimento definitivo usa Covenant-72B
- A combinação SparseLoCo+LoRA é inédita na literatura: LoRA resolve eficiência de parâmetros e SparseLoCo resolve eficiência de comunicação — cenários distintos mas complementares

---

## Referências

- Sarfi et al. (2025). *Communication Efficient LLM Pre-Training with SparseLoCo.* arXiv:2508.15706
- Lidin et al. (2026). *Covenant-72B: Pre-Training a 72B LLM with Trustless Peers Over-the-Internet.*
- Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
- Douillard et al. (2023). *DiLoCo: Distributed Low-Communication Training of Language Models.*
- Jin et al. (2020). *What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams.* arXiv:2009.13081
