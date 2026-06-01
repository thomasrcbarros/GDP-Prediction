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

### Conjuntos de variáveis (A/B/C/D/E)

Os modelos multivariados são testados com diferentes conjuntos de indicadores (os de
atividade entram como **variação % T/T dessazonalizada**, mesma escala do alvo; CAGED e
confiança entram em nível):

| Conjunto | Variáveis | Uso |
|----------|-----------|-----|
| **A** | IBC-Br | baseline (bridge) |
| **B** | PIM + PMS + PMC (indústria, serviços, comércio) | VARX |
| **C** | IBC-Br + PIM + PMS + PMC | VARX |
| **D** | IBC-Br + IPCA + câmbio (bloco macro) | VARX |
| **E** | PIM + PMS + PMC + CAGED + confiança FGV — **sem IBC-Br** | VARX (+ dummy COVID) |

Variantes mantidas (a pedido): **baseline bridge[A]**, **ARIMA/SARIMA** (nos gráficos) e
**apenas os VARX** entre os multivariados — o VAR puro e bridge[B/C] foram removidos.

O **conjunto E** é um VARX de mercado de trabalho + expectativas (CAGED de empregos formais
e Índice de Confiança da Indústria da FGV), **sem o IBC-Br**, e recebe uma **dummy de pico
da COVID (2020Q1-Q2)** como regressor exógeno verdadeiro (não condicionado).

O backtest agora reporta, além do RMSE total: **RMSE pré-2020** e **RMSE pós-2020**
separados, e pode usar **janela rolante de estimação** (`--train-window N`, em trimestres;
ex.: `20` ≈ 5 anos) em vez da amostra completa.

> Os números de RMSE serão preenchidos após a execução local (`--model all --backtest`).
> Em rodadas anteriores, a referência ficou em torno de: bridge[A] ~0.45, varx[D] ~0.50,
> varx[C] ~1.3, varx[B] ~1.7, ARIMA/SARIMA ~2.1, random walk ~3.1.

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
| CAGED — saldo de empregos formais | BCB SGS | 28763 ⚠️ | ~30 d | soma trimestral (nível) |
| Confiança da Indústria — ICI/FGV | BCB SGS | 4393 ⚠️ | ~5 d | média trimestral (nível) |

> ⚠️ Os códigos SGS de **CAGED (28763)** e **confiança FGV (4393)** foram identificados
> automaticamente e **devem ser confirmados no teste local** — basta rodar
> `python scripts/fetch_data.py --refresh` e conferir os nomes/valores. Se algum não
> bater, ajuste em `config.SERIES`.

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
#   --set {A,B,C,D,E}                conjunto de variáveis (modelos multivariados)
#   --target {qoq,yoy}               crescimento T/T-1 ou T/T-4
#   --backtest                       roda backtest realista vs look-ahead
#   --train-window N                 janela rolante de estimação (N trimestres; ~20 = 5 anos)
#   --plot                           gera gráfico comparativo (implica --backtest)
#   --plot-file CAMINHO              PNG de saída (padrão: backtest_comparison.png)
#   --refresh                        recoleta os dados

# Baseline (bridge com IBC-Br):
python scripts/run_nowcast.py --model bridge --set A --backtest

# Novo VARX de trabalho+expectativas (sem IBC-Br, com dummy COVID):
python scripts/run_nowcast.py --model varx --set E --backtest

# Janela rolante de 5 anos (20 trimestres) vs amostra completa:
python scripts/run_nowcast.py --model varx --set E --backtest --train-window 20

# Com gráfico comparando todas as variantes:
python scripts/run_nowcast.py --model all --plot
```

> Com `--model all`, o backtest re-treina ARIMA/SARIMA com busca de ordem em cada janela
> — leva vários minutos. Para iterar rápido, rode um modelo/conjunto por vez
> (ex.: `--model varx --set E`).

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
