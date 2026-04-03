-- ============================================================
-- scripts/supabase_setup.sql
-- Execute no Supabase SQL Editor (uma vez, na configuração inicial)
-- FinTax Agents — 5 Agentes Especializados
-- ============================================================

-- 1. Habilita pgvector
create extension if not exists vector;

-- ============================================================
-- 2. Função para criar tabelas de conhecimento por agente
-- ============================================================
create or replace function create_kb_table(tname text)
returns void language plpgsql as $$
begin
  execute format('
    create table if not exists %I (
      id            bigserial       primary key,
      content       text            not null,
      embedding     vector(1536)    not null,
      metadata      jsonb           default ''{}''::jsonb,
      file_name     text,
      file_hash     text,
      folder        text,
      agent         text,
      chunk_index   integer         default 0,
      h1            text            default '''',
      h2            text            default '''',
      h3            text            default '''',
      indexed_at    timestamptz     default now()
    );

    -- Índice vetorial (IVFFlat — ideal até ~1M registros)
    create index if not exists %I on %I
      using ivfflat (embedding vector_cosine_ops) with (lists = 100);

    -- Índice por hash do arquivo (deduplicação no pipeline)
    create index if not exists %I on %I (file_hash, chunk_index);

    -- Índice por agente/pasta
    create index if not exists %I on %I (agent, folder);
  ',
    tname,
    tname || ''_emb_idx'',  tname,
    tname || ''_hash_idx'', tname,
    tname || ''_agt_idx'',  tname
  );
end;
$$;

-- ============================================================
-- 3. Cria as 5 tabelas de conhecimento (uma por agente)
-- ============================================================
select create_kb_table('kb_contabil');    -- Analista Contábil Sênior
select create_kb_table('kb_fiscal');      -- Analista Fiscal Sênior
select create_kb_table('kb_pessoal');     -- Analista Depto. Pessoal Sênior
select create_kb_table('kb_societario'); -- Analista Direito Societário Sênior
select create_kb_table('kb_abertura_ma');-- Analista Abertura Empresas MA

-- ============================================================
-- 4. Função de busca semântica (RAG retrieval)
--    Chamada pelo app/chat.py via supabase.rpc("match_documents")
-- ============================================================
create or replace function match_documents(
  query_embedding  vector(1536),
  match_table      text,
  match_threshold  float   default 0.70,
  match_count      integer default 8
)
returns table (
  id          bigint,
  content     text,
  metadata    jsonb,
  similarity  float
)
language plpgsql as $$
begin
  return query execute format('
    select
      id,
      content,
      metadata,
      1 - (embedding <=> $1) as similarity
    from %I
    where 1 - (embedding <=> $1) > $2
    order by embedding <=> $1
    limit $3
  ', match_table)
  using query_embedding, match_threshold, match_count;
end;
$$;

-- ============================================================
-- 5. Tabela de log de indexações (pipeline)
-- ============================================================
create table if not exists indexing_log (
  id          bigserial       primary key,
  agent       text,
  file_name   text,
  file_hash   text,
  chunks      integer,
  pages       integer,
  status      text,           -- 'ok' | 'skipped' | 'error'
  error_msg   text,
  indexed_at  timestamptz     default now()
);

-- ============================================================
-- 6. Tabela de controle do crawler (4 camadas de deduplicação)
-- ============================================================
create table if not exists crawl_log (
  id              bigserial       primary key,
  url             text            not null,
  url_hash        text            not null unique,   -- SHA-256 da URL
  filename        text            not null,
  folder_name     text            not null,
  drive_file_id   text,
  file_size_kb    integer,
  content_hash    text,                              -- SHA-256 do conteúdo
  status          text            not null,          -- 'downloaded'|'skipped'|'error'
  error_msg       text,
  source_page     text,
  first_seen_at   timestamptz     default now(),
  last_checked_at timestamptz     default now(),
  downloaded_at   timestamptz
);

create index if not exists crawl_log_url_hash_idx    on crawl_log (url_hash);
create index if not exists crawl_log_content_hash_idx on crawl_log (content_hash);
create index if not exists crawl_log_folder_idx      on crawl_log (folder_name);
create index if not exists crawl_log_status_idx      on crawl_log (status);

-- ============================================================
-- 7. Desabilita RLS para service_role (o microserviço usa
--    service_role key — acesso total sem necessidade de RLS)
-- ============================================================
alter table kb_contabil     disable row level security;
alter table kb_fiscal       disable row level security;
alter table kb_pessoal      disable row level security;
alter table kb_societario   disable row level security;
alter table kb_abertura_ma  disable row level security;
alter table indexing_log    disable row level security;
alter table crawl_log       disable row level security;

-- ============================================================
-- 8. Verificação final — deve retornar 5 tabelas kb_*
-- ============================================================
select table_name
from information_schema.tables
where table_schema = 'public'
  and table_name like 'kb_%'
order by table_name;

-- Resultado esperado:
-- kb_abertura_ma
-- kb_contabil
-- kb_fiscal
-- kb_pessoal
-- kb_societario
