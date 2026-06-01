# GDP-Prediction — Nowcasting do PIB Brasileiro

O PIB brasileiro é divulgado pelo IBGE com cerca de dois meses de atraso em relação ao
fim do trimestre. Este projeto estima o crescimento econômico **em tempo real**
(*nowcasting*) usando indicadores mensais antecedentes — abordagem usada por bancos
centrais e instituições financeiras — antes da divulgação oficial.

Implementa **ARIMA**, **SARIMA**, **VAR** e uma **bridge equation** com *backtesting*
rolling e avaliação explícita de **look-ahead bias** (respeitando o calendário real de
publicação de cada série). Dados públicos do **BCB (SGS)** e **IBGE**.

### Modelos e por que a bridge equation importa

ARIMA/SARIMA são univariados (extrapolam o passado do PIB) e o VAR prevê os próprios
indicadores — nenhum deles explora a real vantagem do nowcasting: os indicadores mensais
do trimestre corrente (IBC-Br, produção industrial) que **já foram publicados** quando o
PIB ainda não saiu. A **bridge equation** regride o PIB sobre esses indicadores
contemporâneos e, por isso, é a que de fato supera os baselines.

### Conjuntos de variáveis (A/B/C/D)

Os modelos multivariados são testados com diferentes conjuntos de indicadores (todos
entram como **variação % T/T dessazonalizada**, mesma escala do alvo):

| Conjunto | Variáveis | Uso |
|----------|-----------|-----|
| **A** | IBC-Br | bridge, VAR |
| **B** | PIM + PMS + PMC (indústria, serviços, comércio) | bridge, VAR |
| **C** | IBC-Br + PIM + PMS + PMC | bridge, VAR |
| **D** | IBC-Br + IPCA + câmbio (bloco macro) | VAR |

No backtest (janela de teste comum, treino com histórico completo do PIB), o RMSE fica:

| Variante | RMSE | RMSE sem 2020 |
|----------|------|----------------|
| random walk (baseline) | 3.14 | 0.96 |
| média histórica (baseline) | 2.12 | 0.57 |
| **bridge[A] — IBC-Br** | **0.45** | **0.44** |
| bridge[C] — todos | 0.74 | 0.59 |
| bridge[B] — setorial | 1.56 | 0.90 |
| var[A] / var[D] | ~2.9 / ~3.0 | ~0.6 / ~0.9 |
| var[B] / var[C] | ~5.7 / ~6.0 | ~4.7 / ~4.8 |

**Conclusão empírica:** a **bridge equation usando o IBC-Br (conjunto A)** é disparado o
melhor nowcast — RMSE ~0.45 contra 3.14 do random walk. Faz sentido: o IBC-Br é o próprio
proxy mensal do PIB calculado pelo BCB. O trio setorial (B) sozinho perde para o IBC-Br;
juntar tudo (C) fica entre os dois. O VAR é instável como previsor pontual (prevê os
indicadores em vez de condicionar nos valores já conhecidos). O choque da COVID (2020) é
tratado como **quebra estrutural** via dummies de intervenção, e o RMSE é reportado também
excluindo 2020. (Números variam um pouco a cada coleta de dados.)

## Indicadores

| Série | Fonte | Código | Defasagem | Transformação |
|-------|-------|--------|-----------|---------------|
| PIB (índice dessaz. trimestral) — *alvo* | BCB SGS | 22109 | ~60 d | var % T/T |
| IBC-Br dessazonalizado | BCB SGS | 24364 | ~45 d | var % T/T |
| PIM-PF indústria geral (dessaz) | BCB SGS | 21859 | ~35 d | var % T/T |
| PMS volume serviços (dessaz) | IBGE SIDRA | 8688 / 7168 | ~45 d | var % T/T |
| PMC volume varejo (dessaz) | IBGE SIDRA | 8880 / 7170 | ~45 d | var % T/T |
| IPCA (variação mensal) | BCB SGS | 433 | ~10 d | soma trimestral |
| Câmbio R$/US$ (venda, média) | BCB SGS | 3698 | ~1 d | var % T/T |

As defasagens de publicação (`config.py`) são o núcleo da avaliação de *look-ahead bias*:
permitem simular exatamente quais dados estariam disponíveis numa dada data de referência.

## Instalação

```bash
pip install -r requirements.txt
```

## Uso

```bash
# 1. Coleta e cacheia as séries em data/ (use --refresh para forçar nova busca)
python scripts/fetch_data.py

# 2. Gera o nowcast e roda o backtesting
python scripts/run_nowcast.py --model all --backtest
#   --model {arima,sarima,var,bridge,all}   escolhe o(s) modelo(s)
#   --set {A,B,C,D}                  conjunto de variáveis (modelos multivariados)
#   --target {qoq,yoy}               crescimento T/T-1 ou T/T-4
#   --backtest                       roda backtest realista vs look-ahead
#   --plot                           gera gráfico comparativo (implica --backtest)
#   --plot-file CAMINHO              PNG de saída (padrão: backtest_comparison.png)
#   --refresh                        recoleta os dados

# Melhor modelo isolado (bridge com IBC-Br):
python scripts/run_nowcast.py --model bridge --set A --backtest

# Com gráfico comparando todas as variantes:
python scripts/run_nowcast.py --model all --plot
```

> Com `--model all`, o backtest re-treina ARIMA/SARIMA com busca de ordem em cada janela
> e roda 9 variantes — leva vários minutos. Para iterar rápido, rode um modelo/conjunto
> por vez (ex.: `--model bridge --set A`).

O `--plot` salva um PNG com três painéis: (1) previsões *one-step-ahead* de cada
modelo vs. PIB observado no período de backtest; (2) RMSE por modelo (regime
realista); (3) nowcast do próximo trimestre por modelo.

Saída típica do backtest: uma tabela comparando, por modelo, o erro no regime
**realista** (janela expansível, *one-step-ahead*) com o regime **look-ahead** (ajuste
sobre a amostra inteira). A diferença quantifica o viés de antecipação — o regime
look-ahead apresenta erros sistematicamente menores justamente porque vaza informação
do futuro.

## Estrutura

```
config.py                      # códigos das séries, defasagens, parâmetros
src/gdp_nowcast/
├── data_sources.py            # clientes BCB SGS + IBGE (com cache CSV)
├── dataset.py                 # alinhamento mensal→trimestral, alvo, anti look-ahead
├── preprocessing.py           # ADF, diferenciação reversível
├── models/{arima,sarima,var,bridge}.py
├── backtest.py                # rolling-origin, métricas, look-ahead
└── nowcast.py                 # pipeline ponta-a-ponta
scripts/{fetch_data,run_nowcast}.py
tests/                         # pytest
```

## Testes

```bash
python -m pytest
```

Cobrem o alinhamento trimestral, a reversibilidade das transformações, a guarda
anti look-ahead (asserção de que dados posteriores à data de referência não vazam) e o
ajuste/forecast dos três modelos.

## Avisos

- Os dados em `data/*.csv` não são versionados (ver `.gitignore`); rode `fetch_data.py`
  para populá-los.
- O nowcast usa séries já dessazonalizadas/revisadas do SGS; uma extensão natural seria
  incorporar *vintages* reais (dados como publicados na época) para um teste de
  look-ahead ainda mais fiel.
