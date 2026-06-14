# SparseLoCo + LoRA — Fine-tuning do Covenant-72B no MedQA

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

O SparseLoCo foi proposto e validado exclusivamente para pré-treinamento com gradientes densos. Este trabalho explora uma aplicação nova: fine-tuning com LoRA do Covenant-72B em um domínio específico (médico), avaliando se a compressão Top-k com error feedback mantém a qualidade comparável ao fine-tuning centralizado.

Comparamos duas abordagens no benchmark MedQA (USMLE):

| Abordagem | Parâmetros treináveis | Comunicação |
|---|---|---|
| AdamW + LoRA (baseline) | ~103M (LoRA rank=8) | centralizado, sem compressão |
| SparseLoCo + LoRA | ~103M (LoRA rank=8) | Top-k 10% comprimido, distribuído simulado |

**Modelo:** Covenant-72B com QLoRA 4-bit NF4 (bfloat16)
**Benchmark:** MedQA — questões médicas USMLE (A/B/C/D/E, chance=20%)
**Workers simulados:** R=4, executados em série na mesma máquina
**Hardware:** A100 80GB (recomendado)

---

## Configuração do Experimento

| Parâmetro | Valor | Justificativa |
|---|---|---|
| `B` | 1 | Covenant-72B ocupa ~36GB, pouca VRAM restante |
| `MAX_LEN` | 512 | Cobre a maioria das questões MedQA sem OOM |
| `LORA_R` | 8 | EF buffers gerenciáveis (~1.6GB para 4 workers) |
| `LR_INNER` | 1e-4 | Conservador para modelo de 72B |
| `H` | 20 | Inner steps por round |
| `T` | 10 | Outer rounds (~1h cada no A100) |
| `TOPK_K` | 0.10 | 90% de compressão nos pseudo-gradientes |
| `BETA` | 0.95 | Decay do error feedback |
| `MAX_GRAD_NORM` | 1.0 | Gradient clipping para estabilidade |

---

## Estrutura

```
sparseloco-with-lora/
└── sparseloco-with-lora.ipynb    # notebook principal
```

---

## Como Executar

### Dependências

```bash
!pip install torch transformers peft datasets matplotlib tqdm scikit-learn bitsandbytes>=0.46.1 accelerate
```

### Hardware necessário

| Modelo | VRAM mínima |
|---|---|
| Covenant-72B (4-bit) | A100 80GB |
| TinyLlama-1.1B (fallback) | T4 16GB |

### Executar

1. Abrir `sparseloco-with-lora.ipynb` no Google Colab com **A100 GPU**
2. Montar o Google Drive (célula de config já faz isso automaticamente)
3. Rodar todas as células em ordem

Os checkpoints são salvos automaticamente no Google Drive após cada round, permitindo retomar após desconexão.

---

## Limitações

- O SparseLoCo é simulado localmente — workers rodam em série na mesma máquina, sem comunicação real pela rede
- Com B=1 e T=10, o experimento é um subset do que seria ideal — limitação de tempo de GPU
- A motivação prática combina SparseLoCo (eficiência de comunicação) com LoRA (eficiência de parâmetros) — cenários distintos, mas a combinação é inédita na literatura

---

## Referências

- Sarfi et al. (2025). *Communication Efficient LLM Pre-Training with SparseLoCo.* arXiv:2508.15706
- Lidin et al. (2026). *Covenant-72B: Pre-Training a 72B LLM with Trustless Peers Over-the-Internet.*
- Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
- Douillard et al. (2023). *DiLoCo: Distributed Low-Communication Training of Language Models.*
- Jin et al. (2020). *What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams.* arXiv:2009.13081
