# Projeto: optcg-knowledge — Servidor MCP de regras, cartas e rulings do One Piece Card Game

## Objetivo
Criar um servidor MCP em Python que funcione como base de conhecimento do One Piece Card Game (OPTCG):
regras, cartas, FAQs oficiais e rulings de juízes, com busca híbrida.
O servidor NÃO chama nenhuma LLM: ele só expõe dados e ferramentas. Quem raciocina é o agente conectado
(Claude Code, Antigravity, Gemini CLI).

Casos de uso:
1. Juiz de regras: responder dúvidas citando regra/FAQ/ruling, respeitando a hierarquia de autoridade.
2. Curadoria: transformar respostas de juízes (exportadas de chat) em rulings estruturados.
3. Fonte de dados para outros servidores (ex.: um futuro `optcg-engine` MCP, que vive em outro repositório
   e consome cartas e rulings daqui).

## Fora do escopo
- Engine de jogo, parser de efeitos, DSL, validação e simulação. Isso fica em um servidor MCP separado,
  no repositório da engine. Este projeto não importa, não altera e não depende da engine.
- Projete pensando em interoperabilidade: IDs de carta estáveis (`OP05-119`), IDs de regra estáveis (`7-1-4-1`)
  e export de rulings em JSON.

## Restrições
- Python 3.12, gerenciado com `uv` (pyproject.toml). Transporte MCP via stdio usando o SDK oficial `mcp` (FastMCP).
- Zero chamadas a APIs de LLM ou de embedding pagas. Embeddings LOCAIS com sentence-transformers
  (modelo multilíngue: `BAAI/bge-m3` ou `intfloat/multilingual-e5-base`; deixe configurável).
- Banco: um único SQLite com FTS5 (BM25) + `sqlite-vec` (vetores).
- Ingestão idempotente (pode rodar de novo sem duplicar dados).
- Não baixar nem hospedar imagens de cartas (propriedade da Bandai). Só texto e stats; imagem apenas como URL.
- Rulings de juízes: guardar só o conteúdo e o papel ("judge"), nunca nomes de pessoas.
- Não invente URLs de fontes. Se uma fonte de dados não for encontrada ou acessível, deixe a ingestão
  plugável e aceite arquivos manuais em `data/raw/`.

## Hierarquia de autoridade (central no projeto)
| authority | source                | descrição                                  |
|-----------|-----------------------|--------------------------------------------|
| 1         | rules                 | Comprehensive Rules                        |
| 2         | official_faq / errata | FAQs oficiais por carta e erratas          |
| 3         | judge                 | Respostas de juízes (grupo oficial Bandai) |
| 4         | community             | Comunidade (opcional)                      |

Rulings podem ficar desatualizados: guarde `date` e `rules_version`, e marque como `superseded_by` em vez de apagar.

## Esquema do banco (ajuste se necessário, justificando)
- `rules(rule_id PK  -- ex "7-1-4-1", parent_id, section, text, rules_version, effective_date)`
- `cards(card_id PK  -- ex "OP05-119", name, set_code, category, colors, cost, power, counter,
   attributes, traits, effect_text, trigger_text, rarity, image_url, raw_json)`
- `rulings(ruling_id PK, question, answer, source, authority, date, rules_version, url,
   superseded_by, reviewed BOOL, raw_ref)`
- `ruling_cards(ruling_id, card_id)` e `ruling_rules(ruling_id, rule_id)`
- `chunks(chunk_id PK, doc_type  -- rule|ruling|card, doc_id, text, authority, metadata JSON)`
  + tabela FTS5 sobre `chunks.text` + tabela vetorial sqlite-vec
- `meta(key, value)` para versão das regras, modelo de embedding e data de ingestão

Chunking: um chunk por regra (incluindo o caminho dos pais no texto do chunk para dar contexto),
um chunk por ruling (pergunta e resposta juntas), um chunk por carta (nome, tipo, efeito, trigger).

## Busca
Busca híbrida: BM25 (FTS5) + vetorial, combinadas com Reciprocal Rank Fusion. Boost leve por
autoridade (as mais altas primeiro em caso de empate). Detecte na query IDs de carta (`OP05-119`, `ST10-001`),
números de regra (`7-1-4`) e keywords entre colchetes (`[Blocker]`, `[On K.O.]`), e priorize match exato.
Por padrão, ignore rulings `superseded`.

## Tools MCP
Consulta:
- `search(query, k=8, doc_types=None, min_authority=None, include_superseded=False)`
- `get_rule(rule_id, include_children=True)`
- `get_card(id_or_name)` → aceita ID exato ou nome (fuzzy), retorna todas as versões se houver várias
- `search_cards(name=None, color=None, cost=None, trait=None, keyword=None, set_code=None)`
- `get_card_rulings(card_id)` → ordenado por autoridade e data

Curadoria:
- `add_ruling(question, answer, source, date, card_ids, rule_refs, url=None, raw_ref=None)` → entra com `reviewed=false`
- `mark_ruling(ruling_id, reviewed=None, superseded_by=None)`
- `list_unreviewed_rulings(limit=20)`

Interoperabilidade:
- `export_rulings(card_id=None, since=None, min_authority=None)` → JSON estável (também disponível via CLI),
  para outros projetos (ex.: engine) gerarem testes de regressão a partir dos rulings.

## Prompts MCP (fluxos expostos pelo servidor)
- `juiz`: consultar `search` / `get_card` / `get_card_rulings`; responder pela fonte de maior autoridade;
  sempre citar regra/ruling (ID e fonte); se houver conflito entre fontes, explicitar; se o ruling for
  anterior à versão atual das regras, avisar. Se houver um servidor de engine conectado com uma tool de
  simulação, pode usá-la para confirmar interações, mas o prompt deve funcionar sem ela.
- `curar(arquivo)`: ler uma exportação de chat em `data/raw/judge_chats/`, extrair pares pergunta/resposta
  de juízes, remover nomes, ligar a cartas e regras, e chamar `add_ruling`.

## Ingestão (CLI: `optcg-ingest`)
- `rules <pdf>`: parsear o PDF das Comprehensive Rules em regras hierárquicas por número.
- `cards [--source ...]`: importar cartas de uma fonte JSON comunitária (pesquise uma fonte confiável e
  me mostre antes de usar) ou de arquivos manuais em `data/raw/cards/`.
- `faq <pdf|dir>`: parsear FAQs/erratas oficiais por coleção em rulings (authority 2).
- `embed`: (re)gerar embeddings dos chunks novos ou alterados.
- `export-rulings [--out arquivo.json]`.

## Testes
pytest para: parsing de regras, parsing de FAQ, busca híbrida (queries de exemplo com resultado esperado,
incluindo match exato de ID de carta, número de regra e keyword) e tools MCP.

## Documentação
README com: instalação, ingestão, e como conectar o servidor no Claude Code (`claude mcp add`) e no Antigravity.

## Forma de trabalho
Trabalhe em fases e PARE ao fim de cada uma para eu revisar:
1. Estrutura do projeto, schema, ingestão de regras e cartas, testes básicos.
2. Busca híbrida + tools de consulta + prompt `juiz`.
3. FAQs oficiais (authority 2).
4. Curadoria de rulings de juízes + prompt `curar` + `export_rulings`.

Se algo for ambíguo ou depender de uma decisão minha (fonte de dados, formato, nomes), pergunte em vez de assumir.
