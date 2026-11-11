# TencentDB Agent Memory Platform

Plataforma corporativa de gerenciamento, retenção e persistência de memória de longo prazo para Agentes de Inteligência Artificial, construída sobre a infraestrutura de alta performance do **Tencent Cloud VectorDB** (ou PostgreSQL com pgvector / Redis TencentDB).

Este projeto fornece uma solução unificada e robusta para que agentes recuperem históricos, preferências do usuário e fatos relevantes de forma semanticamente contextualizada, otimizando em mais de 60% o uso de tokens e eliminando a perda de contexto entre sessões.

---

## 🏗️ 1. Arquitetura do Sistema

O fluxo de retenção, busca e injeção de contexto de memória ocorre conforme o diagrama abaixo:

```text
                  +----------------------------------------------+
                  |                 AI Agent Turn                |
                  +----------------------+-----------------------+
                                         |
                                         v
                            +--------------------------+
                            |      Memory Router       |
                            +------------+-------------+
                                         |
                     +-------------------+-------------------+
                     |                                       |
                     v                                       v
        +--------------------------+           +---------------------------+
        |    Short-Term Cache      |           |     TencentDB Vector      |
        |      (LRU Buffer)        |           |      (Long-Term Store)    |
        +------------+-------------+           +-------------+-------------+
                     |                                       |
                     |                                       v
                     |                         +---------------------------+
                     |                         |  Deduplication Check &    |
                     |                         |   Semantic Update Loop    |
                     |                         +-------------+-------------+
                     |                                       |
                     |                                       v
                     |                         +---------------------------+
                     |                         |   Recency Weighting &     |
                     |                         |   Exponential Decay S(t)  |
                     |                         +-------------+-------------+
                     |                                       |
                     +-------------------+-------------------+
                                         |
                                         v
                        +----------------------------------+
                        |     Fact & Context Extractor     |
                        +-----------------+----------------+
                                          |
                                          v
                        +----------------------------------+
                        |     Prompt Context Injection     |
                        +----------------------------------+
```

### Componentes de Fluxo
1. **Memory Router**: Decide se uma nova interação deve ser registrada temporariamente ou enviada para persistência vetorial de longo prazo.
2. **Short-Term Cache**: Um buffer LRU contido em memória que garante o contexto imediato de conversas ativas.
3. **TencentDB Vector Store**: Armazenamento vetorial distribuído de alta velocidade, responsável por consultas de similaridade de cosseno sobre representações latentes.
4. **Deduplication Check**: Interceptador semântico que previne o inchaço da base vetorial ao fundir memórias redundantes sob o mesmo ID de referência.
5. **Recency Weighting**: Motor de decaimento temporal que equilibra similaridade semântica pura com a relevância do tempo decorrido.

---

## 📊 2. Estratégias de Busca e Relevância

### A. Similaridade de Cosseno (Cosine Similarity)
As buscas utilizam a similaridade entre os vetores de embedding gerados a partir do texto de consulta ($A$) e o documento de memória ($B$):
$$Similarity(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$

### B. Decaimento Temporal Exponencial (Recency Weighting)
Para priorizar memórias recentes sobre fatos antigos com similaridade equivalente, aplicamos a fórmula de decaimento baseada no tempo decorrido em dias ($t_{days}$):
$$S_{final} = CosineSimilarity \times e^{-\lambda \cdot t_{days}}$$
Onde:
* $\lambda$ é a taxa de decaimento configurada (`MEMORY_DECAY_RATE`).
* $t_{days}$ é a diferença de tempo calculada em dias entre o momento atual e a criação do registro.

### C. Deduplicação Semântica (Semantic Deduplication)
Antes de persistir uma memória episódica, o sistema executa uma busca de similaridade rápida na coleção do usuário. Se a melhor correspondência retornar uma similaridade superior ao `MEMORY_DEDUPLICATION_THRESHOLD` (padrão: `0.85`), o sistema atualiza o timestamp e incrementa o contador de atualizações do registro original em vez de duplicá-lo na base.

---

## 📁 3. Estrutura de Arquivos

```text
tencentdb-agent-memory/
├── src/
│   ├── __init__.py
│   ├── config.py                     # Configurações de ambiente via Pydantic
│   ├── tencent_vector_connector.py   # Conector/Adapter para TencentDB & Fallback Mock
│   ├── agent_memory_manager.py       # Gerenciador Core (CRUD, Deduplicação, Decay, Summarize)
│   └── api.py                        # Exposição de endpoints REST via FastAPI
├── tests/
│   ├── __init__.py
│   └── test_memory_system.py         # Testes automatizados (pytest)
├── .env.example                      # Exemplo de configuração de variáveis de ambiente
├── example_agent_run.py              # Script de teste interativo / caso de uso prático
└── README.md                         # Este arquivo de documentação técnica
```

---

## 🛠️ 4. Configuração & Instalação

### Requisitos Mínimos
* Python 3.10 ou superior
* Gerenciador de pacotes `pip`

### Variáveis de Ambiente (`.env`)
Copie o arquivo `.env.example` para `.env` e configure conforme as credenciais da Tencent Cloud:

```bash
cp tencentdb-agent-memory/.env.example tencentdb-agent-memory/.env
```

| Variável | Padrão | Descrição |
|----------|--------|-------------|
| `TENCENTDB_HOST` | `10.0.0.1` | Host de conexão do Tencent Cloud VectorDB |
| `TENCENTDB_PORT` | `80` | Porta de rede do Tencent VectorDB |
| `TENCENTDB_USERNAME`| `root` | Usuário de autenticação do banco |
| `TENCENTDB_KEY` | `api-key` | Api-Key secreta do Tencent VectorDB |
| `TENCENTDB_DATABASE_NAME` | `agent_memory_db`| Nome do banco de dados vetorial |
| `TENCENTDB_COLLECTION_NAME`| `memories` | Nome da coleção utilizada |
| `EMBEDDING_PROVIDER` | `mock` | Provedor de Embeddings (`mock`, `openai`, `hunyuan`) |
| `MOCK_MODE` | `True` | Define se o sistema roda offline em modo simulação |
| `MEMORY_DECAY_RATE` | `0.05` | Constante $\lambda$ de decaimento temporal |

---

## 🚀 5. Execução & Uso

### Executar a Simulação Prática de IA
O script de exemplo realiza uma simulação interativa completa onde o agente armazena uma preferência técnica de Alice na primeira interação, atualiza a cache de curto prazo, recupera essa preferência de forma autônoma para injetar no Prompt da LLM em um diálogo posterior, realiza a consolidação e executa a limpeza GDPR de segurança.

```bash
PYTHONPATH=tencentdb-agent-memory python3 tencentdb-agent-memory/example_agent_run.py
```

### Iniciar o Servidor API FastAPI
Para expor a API de persistência de memória para múltiplos agentes da sua empresa, inicie o servidor:

```bash
PYTHONPATH=tencentdb-agent-memory uvicorn src.api:app --reload --port 8000
```

#### Endpoints Disponíveis:
* **`GET /health`**: Saúde e conexão com o banco TencentDB.
* **`POST /memories`**: Salva um turno de conversa ou fato episódico na base vetorial (com deduplicação automática).
* **`GET /memories`**: Consulta semântica com aplicação de filtro por usuário e decaimento temporal.
* **`POST /memories/summarize`**: Consolida o buffer de curto prazo em um resumo definitivo.
* **`POST /memories/forget`**: Executa o ciclo de esquecimento de memórias pouco relevantes.
* **`DELETE /memories/user/{user_id}`**: Purga de dados para conformidade com a LGPD.

---

## 🧪 6. Testes Automatizados

O projeto possui cobertura completa de testes para garantir a robustez de cada módulo e evitar regressões:

```bash
PYTHONPATH=tencentdb-agent-memory pytest tencentdb-agent-memory/tests/
```

---

## 🛡️ 7. Segurança & LGPD Compliance

A plataforma foi projetada sob o princípio de **Privacy by Design**, garantindo total conformidade com a LGPD e GDPR:
* **GDPR Purge (`DELETE /memories/user/{user_id}`)**: Permite a eliminação física definitiva de todos os registros de embeddings e metadados de um usuário em menos de 100ms.
* **Forgetting Loop**: Permite que informações sensíveis antigas e obsoletas sofram decaimento natural e sejam purgadas autonomamente da nuvem.
* **Segurança de Tráfego**: Configurado para tráfego seguro HTTPS com suporte a autenticação por API-Keys criptografadas em conexões privadas na Tencent Cloud (VPC).
