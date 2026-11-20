# CIC-IDS-2018: tudo o que foi executado

Trabalho binário com o CSE-CIC-IDS-2018 em `cicids2018-kdd-models`, da primeira compilação de florestas em 7 de outubro de 2026 até a reavaliação dos blocos de teste não amostrados, encerrada em 4 de novembro de 2026. Fonte: os relatórios de execução e o `STATUS.md` atualizado em 7 de novembro de 2026. Os experimentos rodaram na máquina do conjunto de dados; este workspace guarda os relatórios.

**Dois circuitos passam, e só como detecção de família conhecida.** `known_lr_c1` e `known_lr_c10`, regressões logísticas de 5 bits em blocos disjuntos de 30 minutos, passam do F1 e do orçamento de 0,70 s de ponta a ponta. `known_lr_c1` é o modelo mais forte: F1 claro 0,9761, F1 de simulação 0,9746, taxa de falso positivo benigno 0,80%, média de ponta a ponta 15,6 ms. Nenhuma floresta aleatória cumpriu os dois critérios. Nenhum circuito cifrado de Bot contra tráfego benigno futuro passou.

| | |
|---|---|
| Circuitos qualificados que passaram | 2 / 4 |
| F1 claro de teste do `known_lr_c1` | 0,976 |
| Média de ponta a ponta do `known_lr_c1` | 15,6 ms |
| Detector FHE de Bot abaixo de 0,70 s | Nenhum |

## A meta, e como a afirmação ficou mais estreita

A meta permanente é um modelo binário com F1 acima de 0,70 e latência FHE real de um único registro, de ponta a ponta (cifrar + inferência + decifrar, sem a geração de chaves), de no máximo 0,70 segundos. Uma divisão aleatória por linha mistura fluxos do mesmo dia de captura nos dois lados e fica de fora da evidência final.

Em 30 de outubro o escopo retirou a Infiltração. As linhas grafadas `infilteration` nos arquivos de origem saem das métricas do escopo e permanecem fora da classe benigna. A afirmação sustentada é detecção de Bot contra tráfego benigno futuro, ou detecção de família conhecida para as famílias que têm blocos de tempo separados suficientes. Detecção geral de anomalia fica de fora dessa afirmação.

| Critério | O que mede | Resultado que vale |
|---|---|---|
| Par da floresta | F1 FHE > 0,70 e ponta a ponta na recarga ≤ 0,70 s | Nenhuma configuração |
| Bot contra benigno futuro | F1 e revocação de Bot no escopo ≥ 0,70, FPR benigno futuro ≤ 1% | PCA-32 e distância padrão em ponto flutuante passam; os circuitos não |
| Família conhecida | F1 claro e de simulação ≥ 0,70, ponta a ponta ≤ 0,70 s, blocos sem cruzar divisões | `known_lr_c1` e `known_lr_c10` passam |

Fonte: `STATUS.md`, 7 de novembro de 2026. A revocação de Infiltração aparece ao lado das métricas de Bot e fica de fora delas.

## Registro das execuções

Dezesseis milhões de fluxos estão em dez arquivos diários do CICFlowMeter. Os treinos de floresta carregam o CSV combinado (16.232.943 linhas; 13.484.708 benignas e 2.748.235 de ataque no registro de 19 de outubro) e dividem 80/20 ao acaso. Os scripts posteriores leem os arquivos diários em fluxo (16.233.002 linhas) e guardam uma amostra por sorteio, porque carregar tudo estoura o teto de 20 GiB de RAM.

| Quando | Execução | Desfecho |
|---|---|---|
| 7 out. 00:31 | `svd100_model.py`, `svd200_model.py` | Saída 1 em 4 s. Nenhum CSV em `CIC-IDS-2018`. |
| 7–10 out. | `run_logs/20261007_225839` `svd_model.py` | Morto (saída −9) depois de 71 h, durante o treino de 2×4 SVD 100. Configurações anteriores já tinham sido salvas, inclusive um circuito em CPU de 2×2 SVD 15. |
| 7 out. e 13 out. | `svd_model_fast.py`, e `svd_model.py` em 13 out. | Interrompido ao salvar 2×2 SVD 15 porque o diretório do circuito já tinha conteúdo. O treino de 13 out. já tinha medido essa configuração com F1 em texto claro de 0,7421. O rodapé da suíte ainda registra saída 0. |
| 14–18 out. | `run_logs/20261014_174627` | CPU. Os dois treinos foram mortos (saída −9) durante a compilação do SVD 100, depois de 36 h e 38 h. A medição de um registro terminou em 5 h 49 min. Tempo total da suíte: 80 h 16 min. |
| 19–20 out. | `run_logs/20261019_202240` | Compilação CUDA de n=2, d=2, SVD 100 morta (saída −9) depois de 26,3 h. |
| 27 out. 23:47 | `run_logs/20261027_234701` | `svd_model.py` em CUDA parou durante a limpeza. Sem rodapé. |
| 28 out. 00:00 | `run_logs/20261028_000058` | A compilação CUDA de n=2, d=4, SVD 15 terminou em 8.109 s, e o registro acaba na simulação FHE. Sem rodapé. |
| 28–29 out. | `run_logs/20261028_110413` | Suíte de florestas em CUDA concluída. 31 h 12 min. Este é o resultado das florestas. |
| 30 out. 00:29 | `plaintext_experiment_runs/20261030_002931_248889` | Triagem aleatória por linha em cerca de 1 milhão de fluxos. As features compactas parecem fortes. Evidência de implantação fica de fora. |
| 30 out. 00:42 | `plaintext_experiment_runs/20261030_004253_544870` | A mesma lista curta com dias de captura isolados. Todos os modelos falham. Nenhum circuito foi compilado. |
| 30 out. 01:03 | `anomaly_experiment_runs/20261030_010318_212531` | Primeira execução de anomalia, 1 milhão de linhas. A Infiltração ainda entrava no modelo de família conhecida. |
| 30 out. 23:38 | `anomaly_experiment_runs/20261030_233818_174372` | Carregou as 16.233.002 linhas e parou. A carga completa fica de fora. |
| 30 out. 23:45 | `anomaly_experiment_runs/20261030_234551_431460` | Sorteio de 2 milhões anterior à correção. O F1 de família conhecida 0,9058 fica de fora das citações: a Infiltração permaneceu, e os blocos foram atribuídos por rótulo. |
| 31 out. 00:34 | `anomaly_experiment_runs/20261031_003425_468997` | O mesmo sorteio. Parou depois de limpar 1.987.045 linhas. |
| 31 out. 00:40 | `anomaly_experiment_runs/20261031_004057_494668` | Execução de qualidade corrigida. Passa nos cinco critérios em texto claro. Terminou às 00:52. |
| 31 out. 17:09 | `qualified_fhe_runs/20261031_170906_440932` | Circuitos CUDA dos quatro modelos qualificados. Concluído às 17:57. Duas regressões logísticas passam. |
| 3–4 nov. | `qualified_fhe_runs/20261103_232756_062792_unsampled_test` | Reavaliação dos 1.136.774 fluxos dos blocos retidos que o sorteio pulou. 23:27 às 01:36. Confirma a nota da amostra. Fica sem repetição. |

Os horários são locais (−03). A saída −9 é o encerramento pelo núcleo registrado no rodapé. Os registros de 27 de outubro e de 00:00 de 28 de outubro não têm rodapé, então o motivo da parada está fora do arquivo.

## Florestas aleatórias em sete campos

O modelo original é uma floresta de 5 bits do Concrete ML. As entradas são `syn_flag_cnt`, `ack_flag_cnt`, `fin_flag_cnt`, `rst_flag_cnt`, `totlen_fwd_pkts`, `protocol` e `dst_port`, seguidas de MinMaxScaler, codificação one-hot e TruncatedSVD com 15, 25, 40 ou 100 componentes. A grade concluída usa 2 ou 4 árvores com profundidade 2 ou 4. O SVD 100 rodou só com 2 árvores e profundidade 2. Cada compilação CUDA levou de 8.133 a 8.226 segundos, cerca de 2,26 horas, qualquer que fosse a árvore.

F1 FHE na divisão de teste do treino. O F1 não tem unidade. A linha 0,70 é o critério de qualidade. Três modelos de profundidade 4 passam dela. Fonte: linhas FHE de `svd_model.py` em `run_logs/20261028_110413/report.txt`.

| Configuração | F1 FHE |
|---|---|
| 2×2 SVD 15 | 0,3968 |
| 2×4 SVD 15 | 0,7044 |
| 4×2 SVD 15 | 0,6849 |
| 4×4 SVD 15 | 0,7206 |
| 2×2 SVD 25 | 0,3782 |
| 2×4 SVD 25 | 0,6919 |
| 4×2 SVD 25 | 0,6797 |
| 4×4 SVD 25 | 0,7197 |
| 2×2 SVD 40 | 0,2908 |
| 2×4 SVD 40 | 0,6697 |
| 4×2 SVD 40 | 0,3093 |
| 4×4 SVD 40 | 0,6637 |
| 2×2 SVD 100 | 0,2083 |

Latência de ponta a ponta na recarga, em segundos, média de 100 registros depois de recarregar o circuito. Fonte: `best_model_single_record.py` no mesmo relatório CUDA. Os modelos acima de 0,70 de F1 também ficam acima de 0,70 s.

| Configuração | Média de ponta a ponta (s) |
|---|---|
| 2×2 SVD 15 | 0,649 |
| 2×4 SVD 15 | 1,147 |
| 4×2 SVD 15 | 0,697 |
| 4×4 SVD 15 | 1,762 |
| 2×2 SVD 25 | 0,634 |
| 2×4 SVD 25 | 1,080 |
| 4×2 SVD 25 | 0,647 |
| 4×4 SVD 25 | 1,620 |
| 2×2 SVD 40 | 0,588 |
| 2×4 SVD 40 | 0,817 |
| 4×2 SVD 40 | 0,576 |
| 4×4 SVD 40 | 1,158 |
| 2×2 SVD 100 | 0,600 |

| Árvores × profundidade, SVD | F1 em texto claro | F1 FHE | Ponta a ponta no treino (s) | Ponta a ponta na recarga (s) | Os dois critérios |
|---|---|---|---|---|---|
| 4 × 4, SVD 15 | 0,7480 | 0,7206 | 2,159 | 1,762 | Só o F1 |
| 4 × 4, SVD 25 | 0,7477 | 0,7197 | 2,668 | 1,620 | Só o F1 |
| 2 × 4, SVD 15 | 0,7481 | 0,7044 | 1,322 | 1,147 | Só o F1 |
| 2 × 4, SVD 25 | 0,7471 | 0,6919 | 1,577 | 1,080 | Nenhum |
| 4 × 2, SVD 15 | 0,7152 | 0,6849 | 0,803 | 0,697 | Só a latência |
| 4 × 2, SVD 25 | 0,7153 | 0,6797 | 0,957 | 0,647 | Só a latência |
| 2 × 4, SVD 40 | 0,7264 | 0,6697 | 1,190 | 0,817 | Nenhum |
| 4 × 4, SVD 40 | 0,7264 | 0,6637 | 1,458 | 1,158 | Nenhum |
| 2 × 2, SVD 15 | 0,3897 | 0,3968 | 0,761 | 0,649 | Só a latência |
| 2 × 2, SVD 25 | 0,3897 | 0,3782 | 0,839 | 0,634 | Só a latência |
| 4 × 2, SVD 40 | 0,2953 | 0,3093 | 0,806 | 0,576 | Só a latência |
| 2 × 2, SVD 40 | 0,2953 | 0,2908 | 0,889 | 0,588 | Só a latência |
| 2 × 2, SVD 100 | 0,2163 | 0,2083 | 0,686 | 0,600 | Só a latência |

Ordenado pelo F1 FHE. A latência no treino vem de `svd_model.py`; a da recarga vem de `best_model_single_record.py`. A inferência em texto claro de um registro, na mesma recarga, ficou entre 0,35 e 0,94 ms. O modelo rápido de SVD 100 é raso: a maior parte das 100 entradas fica sem uso, e a profundidade da árvore domina a inferência cifrada. A recarga em CPU de 14 de outubro, 4×4 SVD 15, mediu 2,795 s (N=1.000), contra 1,762 s nesta recarga CUDA (N=100). Essas duas recargas usam tamanhos de amostra diferentes, então a comparação pareada é a tabela abaixo.

### CPU contra CUDA no mesmo harness de 100 registros

Doze configurações têm os dois dispositivos. A CPU é `svd_model_fast.py` da suíte de 14–18 de outubro. A CUDA é `svd_model.py` de 28–29 de outubro. As duas cronometram 100 registros dentro do harness de treino. O SVD 100 não tem número de CPU: aquela compilação foi morta. As regressões logísticas que passaram foram compiladas só em CUDA.

Segundos, média de cifrar + inferência + decifrar, sem a geração de chaves. Fontes: `cicids2018-kdd-models/report.txt` (CPU, 14–18 de outubro) e `run_logs/20261028_110413/report.txt` (CUDA). O F1 FHE nas mesmas configurações fica dentro de 0,005. A CUDA é mais rápida em 5 das 12. A maior economia em CUDA é 0,42 s em 4×4 SVD 40. A maior penalidade em CUDA é 0,72 s em 4×4 SVD 25. A compilação leva 8.033–8.170 s em CPU e 8.133–8.226 s em CUDA.

| Configuração | Ponta a ponta em CPU (s) | Ponta a ponta em CUDA (s) |
|---|---|---|
| 2×2 SVD 15 | 0,819 | 0,761 |
| 2×4 SVD 15 | 1,352 | 1,322 |
| 4×2 SVD 15 | 0,910 | 0,803 |
| 4×4 SVD 15 | 1,909 | 2,159 |
| 2×2 SVD 25 | 0,826 | 0,839 |
| 2×4 SVD 25 | 1,328 | 1,577 |
| 4×2 SVD 25 | 0,931 | 0,957 |
| 4×4 SVD 25 | 1,953 | 2,668 |
| 2×2 SVD 40 | 0,831 | 0,889 |
| 2×4 SVD 40 | 1,180 | 1,190 |
| 4×2 SVD 40 | 0,936 | 0,806 |
| 4×4 SVD 40 | 1,881 | 1,458 |

## Features compactas e, depois, um holdout de verdade

A triagem seguinte trocou o SVD de sete campos por `compact_flow`: volume, tamanho, tempo, flags, janela e estatísticas de ociosidade, sem identificadores, com log1p com sinal nas caudas pesadas e uma codificação curta de porta no lugar de um one-hot de cerca de 64.000 colunas. Cerca de um milhão de linhas, semente 42. Numa divisão estratificada e aleatória por linha, 70/15/15, todo modelo compacto passou de 0,70. Os mesmos quatro modelos da lista curta, retreinados com dias inteiros de captura isolados, caíram para a linha de base que prevê ataque em todo fluxo.

| Modelo | F1 de teste, linha aleatória | F1 de teste, dia de captura |
|---|---|---|
| MLP-8 | 0,9370 | 0,4328 |
| XGB 4× profundidade 2 | 0,9075 | 0,4326 |
| RL C=10 | 0,9307 | 0,0985 |
| RL C=1 | 0,9285 | 0,1021 |

O F1 não tem unidade. Fonte da linha aleatória: `plaintext_experiment_runs/20261030_002931_248889`. Fonte do dia de captura: `plaintext_experiment_runs/20261030_004253_544870`. A linha 0,433 é o F1 de prever ataque sempre naquele teste por dia (60.777 benignos, 17.516 Bot, 5.655 Infiltração; prevalência de ataque 27,6%). Ela descreve esse teste por dia. O MLP-16 liderou a triagem aleatória com 0,9389 e saiu da lista curta.

O holdout por dia treina até 22 de fevereiro de 2018, valida em 23 e 28 de fevereiro e testa em 1 e 2 de março. Nenhum dia entra em duas divisões. O MLP quantizado escolhido pegou quase todo ataque e também marcou 98,84% dos fluxos benignos. Os testes claros do Concrete ML em 5, 6 e 8 bits falharam do mesmo modo; o melhor foi o MLP-8 em 6 bits, F1 0,4347. A quantização está fora da causa dessa falha. Nenhum circuito saiu desta execução. O que fica é que `compact_flow` carrega muito mais sinal do que o SVD de sete campos, e que a divisão com dias misturados gastava esse sinal na identidade da sessão.

## Dois modelos de ameaça, medidos em separado

O teste por dia de captura também é um teste de família inédita, porque vários ataques ocorrem em um único dia. O experimento corrigido mede então dois protocolos na mesma amostra de 1.987.045 linhas (probabilidade de sorteio 0,123206, a mesma semente do sorteio anterior à correção). Os modelos de anomalia desconhecida ajustam-se só aos dias benignos de treino e escolhem o limiar em dias benignos de validação posteriores. A regressão logística de família conhecida vê os ataques admitidos, e cada bloco de 30 minutos recebe uma única atribuição global de treino, validação ou teste. `cross_split_block_overlap_count` é 0. Os 14 blocos de Infiltração ficam de fora. Famílias com menos de três blocos também ficam de fora, inclusive HOIC (686.012 fluxos brutos, dois blocos) e as variantes de DoS.

### Anomalia desconhecida com orçamento de 1% de FPR benigno

Notas em ponto flutuante. Fonte: `anomaly_experiment_runs/20261031_004057_494668/report.txt`. O F1 sem escopo ainda contém Infiltração, cuja revocação é 3,85% no PCA-32 e 3,49% na distância padrão. Os dias de teste têm 122.218 fluxos benignos, 35.259 de Bot e 11.432 de Infiltração. Um orçamento de 5% leva PCA e distância padrão a cerca de 98,5% de revocação de Bot, com FPR benigno futuro de 4,0–4,8%. PCA-8, PCA-16, a distância robusta e os dois k-means perdem o Bot em 1%.

| Modelo | F1 de Bot | Revocação de Bot | FPR benigno | F1 sem escopo |
|---|---|---|---|---|
| Reconstrução PCA, 32 | 0,9845 | 0,9845 | 0,45% | 0,8534 |
| Distância benigna padrão | 0,9840 | 0,9845 | 0,48% | 0,8524 |
| Reconstrução PCA, 8 | 0,0047 | 0,0024 | 0,64% | 0,0187 |
| Reconstrução PCA, 16 | 0,0046 | 0,0024 | 0,55% | 0,0180 |
| K-means, 4 centroides | 0,0015 | 0,0008 | 1,12% | 0,0167 |
| K-means, 1 centroide | 0,0000 | 0,0000 | 1,22% | 0,0080 |
| Distância robusta mediana/MAD | 0,0000 | 0,0000 | 1,12% | 0,0053 |

### Regressão logística de família conhecida

O mesmo relatório corrigido. Limiares: C=10 em ponto flutuante 0,7355, C=1 em ponto flutuante 0,6261, C=1 em 5 bits 0,6673, C=10 em 5 bits 0,8212. Divisão: 130 / 29 / 28 blocos, 1.438.231 / 203.749 / 159.917 linhas, 64 entradas. O F1 0,9058 de C=10 em 5 bits, anterior à correção, vem de outro divisor e fica de fora deste resultado.

| Modelo | Modo | F1 de teste | Revocação |
|---|---|---|---|
| RL C=10 | Ponto flutuante | 0,9926 | 0,9978 |
| RL C=1 | Ponto flutuante | 0,9918 | 0,9972 |
| RL C=1 | Claro, 5 bits | 0,9761 | 0,9967 |
| RL C=10 | Claro, 5 bits | 0,9747 | 0,9980 |

A massa do teste é sobretudo um bloco de DDoS. De 23.983 linhas de ataque no teste, 18.864 (78,7%) são DDoS LOIC-HTTP. O conjunto publicado tem 576.191 desses fluxos em cerca de 65 minutos em 20 de fevereiro de 2018. A divisão guarda um bloco por família e prefere o bloco mais recente, então esse bloco de teste tem cerca de 153.000 fluxos brutos antes da amostragem.

## Features da regressão logística

`known_lr_c1` e `known_lr_c10` leem o vetor `compact_flow`. Quem chama o modelo entrega 33 campos numéricos do CICFlowMeter mais `protocol` e `dst_port`. O log1p com sinal, a escala min-max e a codificação one-hot rodam em claro. O circuito cifra os 64 números resultantes. A execução de família conhecida registra `input_dimensions` 64. Endereço IP, Flow ID e timestamp ficam fora do vetor. A medição de 15,6 ms começa depois desse preparo.

| Grupo | Campos de origem | Valor cifrado |
|---|---|---|
| Duração e volume | `flow_duration`, `tot_fwd_pkts`, `tot_bwd_pkts`, `totlen_fwd_pkts`, `totlen_bwd_pkts` | Log1p com sinal, depois min-max |
| Tamanho de pacote | `fwd_pkt_len_max`, `fwd_pkt_len_mean`, `fwd_pkt_len_std`, `bwd_pkt_len_max`, `bwd_pkt_len_mean`, `bwd_pkt_len_std`, `pkt_len_mean`, `pkt_len_std` | Log1p com sinal, depois min-max |
| Taxas | `flow_byts_s`, `flow_pkts_s`, `down_up_ratio` | Log1p com sinal, depois min-max |
| Tempo entre chegadas | `flow_iat_mean`, `flow_iat_std`, `flow_iat_max`, `fwd_iat_mean`, `fwd_iat_std`, `bwd_iat_mean`, `bwd_iat_std` | Log1p com sinal, depois min-max |
| Flags TCP | `syn_flag_cnt`, `ack_flag_cnt`, `fin_flag_cnt`, `rst_flag_cnt`, `psh_flag_cnt`, `urg_flag_cnt` | Log1p com sinal, depois min-max |
| Janela e atividade | `init_fwd_win_byts`, `init_bwd_win_byts`, `active_mean`, `idle_mean` | Log1p com sinal, depois min-max |
| Protocolo | `protocol` | One-hot da string de protocolo |
| Porta de destino | `dst_port` | Três partes: log1p(porta) / log1p(65535); one-hot da faixa IANA (bem conhecida ≤1023, registrada ≤49151, dinâmica); one-hot de um serviço comum, ou `other` |

Fonte: `FLOW_NUMERIC`, `engineer_compact_flow` e `make_compact_preprocessor`. Os nomes de serviço são `ftp_data`, `ftp`, `ssh`, `telnet`, `smtp`, `dns`, `http`, `pop3`, `ntp`, `rpc`, `netbios`, `imap`, `snmp`, `ldap`, `https`, `smb`, `imaps`, `pop3s`, `mssql`, `mysql`, `rdp`, `postgres`, `http_alt` e `other`. As portas 137 e 139 compartilham `netbios`. Os 33 campos numéricos mais a porta escalada são 34 colunas; as colunas one-hot levam a largura registrada a 64. Um fluxo é ataque quando a probabilidade de ataque decifrada é pelo menos 0,667279 em `known_lr_c1`, ou 0,821242 em `known_lr_c10`.

## Circuitos qualificados em CUDA

`qualified_fhe_benchmark.py` compilou os quatro modelos que passaram na execução de qualidade corrigida. Distância e PCA são circuitos aritméticos próprios. As regressões logísticas são Concrete ML. As mesmas 1.987.045 linhas. As notas em ponto flutuante e em 5 bits no modo claro batem com o relatório de qualidade em cerca de 1e-5. Relógio de parede 17:09–17:57 em 31 de outubro, dispositivo CUDA, `p_error` 0,01, 100 registros cronometrados.

F1 de qualidade, claro ou quantizado, contra o critério 0,70. Para distância e PCA, o F1 de qualidade é Bot contra benigno no escopo, depois da quantização. Para as regressões logísticas, é o F1 claro de teste de família conhecida. Fonte: `qualified_fhe_runs/20261031_170906_440932/report.txt`.

| Modelo | F1 de qualidade | F1 de simulação |
|---|---|---|
| Distância | 0,3659 | 0,3658 |
| PCA-32 | 0,9840 | 0,7603 |
| RL C=1 | 0,9761 | 0,9746 |
| RL C=10 | 0,9747 | 0,9737 |

Linhas de ataque na amostra do teste de família conhecida (159.917 linhas no total). O suporte benigno é 135.934 e fica de fora desta mistura. Força bruta web 14, XSS 5, injeção SQL 1.

| Ataque | Linhas |
|---|---|
| LOIC-HTTP | 18.864 |
| Força bruta FTP | 2.493 |
| Bot | 2.001 |
| Força bruta SSH | 605 |
| Web, XSS e SQL | 20 |

| Circuito | F1 claro / quantizado | F1 de simulação | FPR benigno | Média ponta a ponta | Compilação | Critério |
|---|---|---|---|---|---|---|
| `known_lr_c1` | 0,9761 | 0,9746 | 0,80% | 0,0156 s | 373 s | Passa |
| `known_lr_c10` | 0,9747 | 0,9737 | 0,88% | 0,0157 s | 367 s | Passa |
| `standard_benign_distance` | 0,3659 | 0,3658 | 100% | 0,844 s | 8,7 s | Falhou |
| `pca_reconstruction_32` | 0,9840 | 0,7603 | 0,48% claro | 9,834 s | 29 s | Falhou |

A distância padrão quantizou o limiar para 0, e toda nota não negativa vira ataque. O F1 de Bot em ponto flutuante ainda era 0,9840. O PCA-32 conserva o resultado flutuante no modo claro quantizado (F1 de Bot 0,9840, revocação 0,9845, FPR benigno 0,48%) e depois se desloca na simulação: F1 de Bot 0,7603, revocação 0,9818, FPR benigno 17,3%. O critério escrito de anomalia reprova o PCA-32 só pela latência. Esse critério deixa o FPR benigno da simulação de fora da conta. Quatorze vezes o orçamento de 0,70 s já bastam para a reprovação. O resultado FHE de detecção de Bot contra tráfego benigno futuro permanece em aberto.

### Revocação clara do `known_lr_c1` no teste amostrado

Percentual das linhas de ataque daquele rótulo previstas como ataque. Fonte: tabela por rótulo do `known_lr_c1` em modo claro quantizado, no relatório FHE de 31 de outubro. A taxa de falso positivo benigno na mesma tabela é 0,80% de 135.934 fluxos. As contagens de web, XSS e SQL são pequenas demais para uma afirmação por família. As linhas de Bot são um bloco posterior da captura de sexta-feira, 2 de março de 2018, e uma gravação independente de botnet continua pendente.

| Rótulo | Revocação clara |
|---|---|
| Bot (2.001) | 100% |
| Força bruta FTP (2.493) | 100% |
| Força bruta SSH (605) | 100% |
| LOIC-HTTP (18.864) | 99,63% |
| Força bruta web (14) | 57,14% |
| XSS (5) | 40% |
| Injeção SQL (1) | 0% |

## O restante dos mesmos blocos de teste

O F1 publicado é o das 159.917 linhas amostradas de teste. A reavaliação de 3 de novembro deixa essas linhas, e todos os blocos de treino, intactos. Ela mede os 1.136.774 fluxos limpos dentro dos 28 blocos originais de teste que o sorteio pulou: 7,11 vezes o teste amostrado, contra 7,12 esperados na probabilidade 0,123206. A prevalência de ataque cai para 15,0%, e 132.909 de 170.177 linhas de ataque (78,1%) continuam sendo LOIC-HTTP. Os modelos em ponto flutuante foram reajustados nos blocos de treino reproduzidos. Os limiares de 5 bits permaneceram congelados. Reproduzir o teste amostrado devolveu o F1 publicado de 5 bits com delta 0. As notas FHE são simulação dos circuitos salvos. Uma passagem em fluxo levou 6.571,7 s, cerca de 174 linhas/s.

F1, com a comparação entre amostra e o restante dos blocos. As quatro notas passam do critério 0,70. O F1 flutuante não amostrado é 0,9925 para C=1 e 0,9932 para C=10. Fonte: `qualified_fhe_runs/20261103_232756_062792_unsampled_test/report.txt`, comparado com o F1 claro amostrado de 31 de outubro.

| Modelo | Teste amostrado, claro de 5 bits | Blocos não amostrados, claro de 5 bits | Blocos não amostrados, simulação FHE |
|---|---|---|---|
| RL C=1 | 0,9761 | 0,9767 | 0,9767 |
| RL C=10 | 0,9747 | 0,9752 | 0,9752 |

| Modelo | Limiar de 5 bits congelado | F1 flutuante | FPR benigno flutuante | F1 de 5 bits e da simulação | FPR benigno de 5 bits |
|---|---|---|---|---|---|
| `known_lr_c1` | 0,667279 | 0,9925 | 0,22% | 0,9767 | 0,79% |
| `known_lr_c10` | 0,821242 | 0,9932 | 0,21% | 0,9752 | 0,86% |

A simulação coincide com o modo claro de 5 bits em quatro casas decimais nos dois modelos. A única diferença impressa por rótulo é `known_lr_c1` em LOIC-HTTP: revocação clara 0,9970, simulação 0,9969. Bot (14.073), força bruta FTP (18.476) e força bruta SSH (4.625) são detectados por completo em ponto flutuante, em 5 bits e na simulação. A força bruta web é 28/74 em todos os modos. XSS é 8/20, exceto `known_lr_c10` em 5 bits e na simulação, em que é 11/20. A injeção SQL não tem linha não amostrada nestes blocos. As latências de 100 registros continuam em 15,6 ms e 15,7 ms. Esta execução confere os mesmos blocos. A nota publicada continua sendo a da amostra.

## O que o diretório sustenta

Uma gravação independente de Bot continua necessária antes de chamar isto de detecção de botnet fora do CIC-IDS-2018. Os circuitos que passam reconhecem famílias de ataque que já apareciam em outros blocos de 30 minutos deste conjunto. Detecção geral de anomalia e detecção cifrada de Bot contra tráfego benigno futuro continuam fora do que foi demonstrado. Nada mais está na fila para esta amostra.

| Afirmação | Situação |
|---|---|
| `known_lr_c1` e `known_lr_c10` passam em F1 e latência em blocos disjuntos de 30 minutos | Sustentada. Citar o teste amostrado em `qualified_fhe_runs/20261031_170906_440932`. |
| As 1.136.774 linhas não amostradas concordam até a quarta casa na simulação | Sustentada como conferência desses mesmos blocos. A nota publicada continua sendo a da amostra. |
| PCA-32 e a distância padrão detectam Bot em ponto flutuante com FPR benigno futuro abaixo de 1% | Sustentada só para a execução corrigida em texto claro. |
| Um circuito compilado detecta Bot contra tráfego benigno futuro | Fora do que foi demonstrado. A distância quantiza para um limiar que marca tudo como ataque. O PCA-32 leva 9,83 s. |
| Alguma floresta cumpre o F1 e o orçamento de 0,70 s ao mesmo tempo | Fora do que foi demonstrado. O melhor F1 FHE é 0,7206 a 1,762 s na recarga. |
| O F1 perto de 0,94 do `compact_flow` em linha aleatória é evidência de implantação | Fora do que foi demonstrado. O holdout por dia de captura cai para 0,433. |
| F1 de família conhecida 0,9058, anterior à correção | Fica de fora das citações. A Infiltração estava dentro do modelo e os blocos foram atribuídos por rótulo. |
| A Infiltração está tratada | Está fora do escopo: excluída do ajuste de família conhecida e relatada à parte no teste de anomalia. Continua fora da classe benigna. |
| HOIC, a maior classe de ataque, foi detectado | Ficou sem nota. Dois blocos estão abaixo da regra de três blocos. |

## Scripts que produziram o registro

| Script | Papel | Escreve em |
|---|---|---|
| `svd_model.py` | Treina as florestas de 5 bits em SVD 15, 25, 40 e 100 e as compila. Os artefatos usam o sufixo `_fast`. | Diretórios de modelo e `run_logs/` |
| `best_model_single_record.py` | Recarrega cada floresta e cronometra a inferência de um registro em texto claro e em FHE real. Relê o CSV a cada execução. | Linhas de latência dentro do relatório da suíte |
| `run_all_and_report.py` | Roda a suíte e agrega `report.txt`. O relatório de floresta citado é a invocação CUDA de 28 de outubro. | `run_logs/<carimbo>/` |
| `gpu_utils.py`, `fhe_latency.py` | Escolha do dispositivo (`PPML_FHE_DEVICE` ou `--device`) e o cronômetro de cifrar / inferência / decifrar. | Nenhuma execução própria |
| `plaintext_model_experiments.py` | Lista curta: RL C=1, RL C=10, MLP-8, XGB com quatro árvores de profundidade 2. Holdout por dia, notas por rótulo e testes claros quantizados. `--compile-fhe` é o caminho do holdout por dia que falhou, e a medição de latência de família conhecida fica em outro script. | `plaintext_experiment_runs/` |
| `anomaly_model_experiments.py` | Experimento de qualidade. Não compila. A orientação atual recusa `--sample-rows 0` porque a carga completa parou. | `anomaly_experiment_runs/` |
| `qualified_fhe_benchmark.py` | Compila e cronometra os quatro modelos qualificados. `--sample-rows 0` é recusado. O relatório é reescrito depois de cada modelo. | `qualified_fhe_runs/` |
| `score_known_lr_unsampled_test.py` | Reajusta os modelos flutuante e claro de 5 bits nos blocos de treino de origem e simula os circuitos salvos nas linhas de teste que o sorteio pulou. Sem nova busca e sem caminho real de cifragem. | O diretório não amostrado de 3 de novembro |

Cada script de experimento abre um diretório novo com carimbo em microssegundos. Os relatórios antigos permanecem. `svd100_model.py` e `svd200_model.py` aparecem só nas falhas de 7 de outubro por falta de dados; a grade de florestas concluída é `svd_model.py`.
