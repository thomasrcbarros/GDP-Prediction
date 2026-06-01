# GDP-Prediction — Nowcasting do PIB Brasileiro

O PIB brasileiro é divulgado pelo IBGE com cerca de dois meses de atraso em relação ao
fim do trimestre. Este projeto estima o crescimento econômico **em tempo real**
(*nowcasting*) usando indicadores mensais antecedentes — abordagem usada por bancos
centrais e instituições financeiras — antes da divulgação oficial.

Implementa **ARIMA**, **SARIMA**, **VAR**, **VARX** (VAR condicional) e uma **bridge
equation** com *backtesting* rolling e avaliação explícita de **look-ahead bias**
(respeitando o calendário real de publicação de cada série). Dados públicos do **BCB
(SGS)** e **IBGE**.

### Modelos

| Modelo | Tipo | Como usa os indicadores |
|--------|------|--------------------------|
| ARIMA / SARIMA | univariado | só o passado do PIB (+ dummies de COVID) |
| VAR | multivariado | sistema conjunto; **prevê** os indicadores |
| VARX | multivariado | VAR conjunto, mas **condiciona** o PIB nos indicadores contemporâneos já publicados |
| bridge | regressão | regride o PIB nos indicadores contemporâneos já publicados |

ARIMA/SARIMA apenas extrapolam o passado do PIB. O VAR prevê os próprios indicadores —
desperdiçando a vantagem do nowcasting. **VARX** e **bridge** exploram a real vantagem: os
indicadores mensais do trimestre corrente (IBC-Br, PIM, PMS, PMC) que **já foram
publicados** quando o PIB ainda não saiu.

### Baseline: IBC-Br (prévia oficial do PIB)

O **IBC-Br** é o índice de atividade econômica do Banco Central — a *prévia oficial* do
PIB. Por isso a **bridge sobre o IBC-Br (conjunto A)** é adotada como **baseline do
projeto**: é o número que qualquer modelo mais elaborado precisa superar para se justificar.

### Conjuntos de variáveis (A/B/C/D)

Os modelos multivariados são testados com diferentes conjuntos de indicadores (todos
entram como **variação % T/T dessazonalizada**, mesma escala do alvo):

| Conjunto | Variáveis | Uso |
|----------|-----------|-----|
| **A** | IBC-Br | baseline (bridge) |
| **B** | PIM + PMS + PMC (indústria, serviços, comércio) | bridge, VAR, VARX |
| **C** | IBC-Br + PIM + PMS + PMC | bridge, VAR, VARX |
| **D** | IBC-Br + IPCA + câmbio (bloco macro) | VARX |

No backtest (janela de teste comum, treino com histórico completo do PIB), o RMSE fica
(números variam um pouco a cada coleta de dados):

| Variante | RMSE | RMSE sem 2020 |
|----------|------|----------------|
| **bridge[A] — IBC-Br (baseline)** | **~0.45** | **~0.44** |
| varx[D] — IBC-Br + IPCA + câmbio | ~0.50 | ~0.41 |
| bridge[C] — todos | ~0.74 | ~0.59 |
| varx[C] — todos | ~1.33 | ~1.13 |
| bridge[B] — setorial | ~1.56 | ~0.90 |
| varx[B] — setorial | ~1.69 | ~1.60 |
| média histórica | ~2.12 | ~0.57 |
| ARIMA / SARIMA | ~2.13 / ~2.13 | ~0.65 / ~0.67 |
| random walk | ~3.14 | ~0.96 |
| VAR[B/C] (prevê indicadores) | ~5.65 / ~5.97 | ~4.68 / ~4.85 |

**Conclusão empírica:** a **bridge sobre o IBC-Br (baseline)** é o melhor nowcast — o
IBC-Br, sendo a prévia oficial do PIB, é quase imbatível como regressor contemporâneo. O
**VARX[D]** (IBC-Br + IPCA + câmbio, condicional) praticamente empata com o baseline e até
o supera fora de 2020. O salto mais importante é VAR → VARX: condicionar nos indicadores já
publicados derruba o RMSE de ~5.7 para ~1.3–1.7 (e ~0.5 no conjunto D), confirmando que o
defeito do VAR puro era prever os indicadores em vez de usá-los. O choque da COVID (2020)
é tratado como **quebra estrutural** via dummies de intervenção, e o RMSE é reportado
também
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
#   --model {arima,sarima,var,varx,bridge,all}  escolhe o(s) modelo(s)
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
├── models/{arima,sarima,var,varx,bridge}.py
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
ajuste/forecast dos modelos (incluindo o condicionamento do VARX).

## Avisos

- Os dados em `data/*.csv` não são versionados (ver `.gitignore`); rode `fetch_data.py`
  para populá-los.
- O nowcast usa séries já dessazonalizadas/revisadas do SGS; uma extensão natural seria
  incorporar *vintages* reais (dados como publicados na época) para um teste de
  look-ahead ainda mais fiel.
