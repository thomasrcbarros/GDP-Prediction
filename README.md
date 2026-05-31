# GDP-Prediction — Nowcasting do PIB Brasileiro

O PIB brasileiro é divulgado pelo IBGE com cerca de dois meses de atraso em relação ao
fim do trimestre. Este projeto estima o crescimento econômico **em tempo real**
(*nowcasting*) usando indicadores mensais antecedentes — abordagem usada por bancos
centrais e instituições financeiras — antes da divulgação oficial.

Implementa **ARIMA**, **SARIMA** e **VAR** com *backtesting* rolling e avaliação
explícita de **look-ahead bias** (respeitando o calendário real de publicação de cada
série). Dados públicos do **BCB (SGS)** e **IBGE**.

## Indicadores

| Série | Fonte | Código | Defasagem de publicação |
|-------|-------|--------|--------------------------|
| PIB (índice dessaz. trimestral) — *alvo* | BCB SGS | 22109 | ~60 dias |
| IBC-Br dessazonalizado | BCB SGS | 24364 | ~45 dias |
| Produção industrial (PIM-PF, indústria geral) | BCB SGS | 21859 | ~35 dias |
| Taxa de desocupação (PNAD Contínua) | BCB SGS | 24369 | ~30 dias |

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
#   --model {arima,sarima,var,all}   escolhe o(s) modelo(s)
#   --target {qoq,yoy}               crescimento T/T-1 ou T/T-4
#   --backtest                       roda backtest realista vs look-ahead
#   --refresh                        recoleta os dados
```

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
├── models/{arima,sarima,var}.py
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
