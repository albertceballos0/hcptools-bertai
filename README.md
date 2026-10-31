# HPC Tools — Lab AI: acelerar BERT-base en SQuAD

Máster en HPC (UDC), asignatura HPC Tools. Fine-tuning de
`google-bert/bert-base-uncased` sobre SQuAD v1.1 (QA extractiva) en PyTorch,
ejecutado en FinisTerrae III (CESGA) con SLURM y GPUs A100.

## Enunciado

> **Lab AI - HPC Tools**
>
> The purpose of this task is to practice the acceleration of the training of a Deep Learning (DL) model written in Pytorch.
>
> Key practical aspects of the task:
> - This task can be done individually or in pairs.
> - The codes have to work in a venv environment like the one provided in the subject.
> - The BASELINE must be tested in one Nvidia A100. The distributed version must be tested in 2 nodes with 2 Nvidia A100 GPUs each.
> - The deliverables have to be submitted by the deadline in a GitHub repository with one folder and one tagged commit per deliverable:
>   - Deliverable 1: Tag: `BASELINE`, Folder name: `BASELINE`
>   - Deliverable 2: Tag: `DISTRIBUTED`, Folder name: `DISTRIBUTED`
> - Each commit must contain:
>   - The Python scripts and the bash and SLURM scripts for the execution.
>   - A README.md file containing:
>     - An explanation of the key contents of the previous files
>     - The reporting of the times and/or the profiling outputs
>     - Any other thing that you would like to comment or clarify about your work
>
> **Deliverable 1: BASELINE - Baseline implementation (Due 30th september 2026)**
>
> Using BERT-Base model (https://huggingface.co/google-bert/bert-base-uncased) and SQUAD dataset (https://rajpurkar.github.io/SQuAD-explorer/), you have to select an implementation in Pytorch for its training using a single GPU. This implementation will be called in the following the BASELINE implementation. In order to generate this implementation you can search for one on the Internet, as the ability to generate such an implementation from scratch is probably beyond your expertise.
>
> You have to measure the training time for that code using one single GPU. If the time is too small (less than one minute), maybe you can add more epochs to the training or look for a larger data set or more sophisticated model architecture.
>
> If you are able to provide a profiling of the training using Tensorboard or any other tool, that will be a plus in your work.
>
> The link to your repository must be submitted by email to diego.andrade@udc.es with the subject "HPC Tool AI Lab Deliverable".
>
> **Deliverable 2: DISTRIBUTED - Distributed implementation (Due 31st october 2026)**
>
> After that, you need to parallelize that code using the native support for Distributed Training in Pytorch or Lightning. The distributed training tool (native or Lightning) and the strategy (DP, DDP, Zero, FSDP) for work distribution depend on you. You can actually experience several if you feel like it.
>
> Once you have the implementation, you have to test it and measure execution times as you did with the BASELINE. Report just times or the output of a profiling time as you did before.
>
> The link to your repository must be submitted by email to diego.andrade@udc.es with the subject "HPC Tool AI Lab Deliverable".

## Entregables

| Entregable | Carpeta | Tag | Contenido |
|---|---|---|---|
| 1 — BASELINE | [`BASELINE/`](BASELINE/) | `BASELINE` | Entrenamiento en 1 GPU A100 |
| 2 — DISTRIBUTED | [`DISTRIBUTED/`](DISTRIBUTED/) | `DISTRIBUTED` | DDP con `torchrun`, 1×2 y 2×2 A100 |

Cada carpeta es autocontenida (scripts Python, bash y SLURM, y su `README.md`
con ficheros clave y tiempos). `main` solo recibe cada entregable cuando está
terminado; el tag marca el commit final de cada uno.

## Cómo reproducir (FT3)

Una vez, en el nodo de login (necesita internet):

```bash
# venv del curso (o cualquier venv con torch + transformers + datasets)
source $STORE/mypython/bin/activate
pip install -r requirements.txt
python BASELINE/prepare_data.py     # modelo y SQuAD en $LUSTRE/hpctools/hf
```

Después, todo por SLURM, nunca en el login:

```bash
sbatch BASELINE/run_baseline.sbatch            # 1 GPU
sbatch DISTRIBUTED/run_1node_2gpu.sbatch       # 1 nodo, 2 GPUs
sbatch DISTRIBUTED/run_ddp.sbatch              # 2 nodos x 2 GPUs
```

Datos, caché de HuggingFace y resultados van a `$LUSTRE/hpctools`, no al
repo ni a `$HOME` (ver `.gitignore`).
