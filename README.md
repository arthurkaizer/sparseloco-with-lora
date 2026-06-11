# SparseLoCo + LoRA — Fine-tuning Distribuído Simulado

Projeto Final do Seminário de Deep Learning.

Aplicação do otimizador **SparseLoCo** — proposto no contexto de pré-treinamento distribuído de LLMs e utilizado no treinamento do Covenant-72B — ao cenário de **fine-tuning com LoRA**, avaliando se a compressão Top-k ainda é eficaz quando os gradientes já são de baixa dimensionalidade por natureza.

---

## Pergunta de Pesquisa

> O SparseLoCo aplicado ao espaço LoRA consegue resultados comparáveis ao baseline centralizado, mesmo descartando 90% dos pseudo-gradientes por round?

---

## Artigos de Base

- **SparseLoCo** — Sarfi et al. (2025). *Communication Efficient LLM Pre-Training with SparseLoCo.* arXiv:2508.15706
- **Covenant-72B** — Lidin et al. (2026). *Pre-Training a 72B LLM with Trustless Peers Over-the-Internet.*

---

## Abordagem

O SparseLoCo foi proposto e validado exclusivamente para pré-treinamento com gradientes densos. Este trabalho explora uma aplicação nova: fine-tuning com LoRA, onde os parâmetros treináveis são ~590K em vez de bilhões. Avaliamos se a compressão Top-k com error feedback consegue manter a qualidade do treinamento nesse regime de baixa dimensionalidade.

Comparamos duas abordagens no benchmark MedMCQA (domínio biomédico):

| Abordagem | Parâmetros treináveis | Comunicação |
|---|---|---|
| AdamW + LoRA (baseline) | LoRA adapters | centralizado, sem compressão |
| SparseLoCo + LoRA | LoRA adapters | Top-k comprimido, distribuído simulado |

**Modelo de teste local:** TinyLlama-1.1B (CPU, FP32, sem quantização)  
**Modelo de desenvolvimento:** Llama-3.2-3B com QLoRA 4-bit (Colab T4, requer login HF)  
**Modelo final:** Covenant-72B com QLoRA 4-bit (requer A100 80GB)  
**Benchmark:** MedMCQA — questões médicas de múltipla escolha (4 classes, chance=25%)  
**Workers simulados:** R=4, executados em série na mesma máquina

---

## Estrutura

```
sparseloco-with-lora/
└── experimento.ipynb    # notebook principal com todos os experimentos
```

---

## Como Executar

### Dependências

```bash
pip install torch transformers peft datasets matplotlib tqdm scikit-learn
```

### Executar

```bash
jupyter notebook experimento.ipynb
```

### Tempo estimado

| Hardware | Tempo |
|---|---|
| CPU | várias horas |
| GPU (T4/RTX) | ~20–40 min |

---

## Limitações

- O SparseLoCo é simulado localmente — workers rodam em série na mesma máquina, sem comunicação real pela rede
- Covenant-72B exige ~36GB de VRAM mesmo com 4-bit quantization — o desenvolvimento local usa TinyLlama-1.1B (CPU, FP32), o intermediário usa Llama-3.2-3B (Colab T4) e o experimento final roda no Colab A100
- A motivação prática da combinação SparseLoCo+LoRA é limitada: LoRA resolve eficiência de memória (problema local) e SparseLoCo resolve eficiência de comunicação (problema distribuído) — são cenários distintos

---

## Referências

- Sarfi et al. (2025). *Communication Efficient LLM Pre-Training with SparseLoCo.* arXiv:2508.15706
- Lidin et al. (2026). *Covenant-72B: Pre-Training a 72B LLM with Trustless Peers Over-the-Internet.*
- Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
- Douillard et al. (2023). *DiLoCo: Distributed Low-Communication Training of Language Models.*
- Wang et al. (2019). *GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding.*
