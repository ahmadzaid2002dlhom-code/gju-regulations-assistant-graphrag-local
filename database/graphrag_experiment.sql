-- Optional future database-backed graph. Do not run this against the submitted
-- production project. The local Python experiment works without this file.

alter table public.sections
    add column if not exists summary_text text;

alter table public.sections
    add column if not exists embedding extensions.vector(768);

create table if not exists public.section_edges (
    id uuid primary key default gen_random_uuid(),
    source_section_id uuid not null references public.sections(id) on delete cascade,
    target_section_id uuid not null references public.sections(id) on delete cascade,
    relation_type text not null check (
        relation_type in (
            'references',
            'defines',
            'exception_to',
            'requires',
            'applies_to',
            'amends',
            'supersedes',
            'translation_of',
            'same_topic',
            'previous_article',
            'next_article'
        )
    ),
    confidence double precision not null default 1.0
        check (confidence between 0 and 1),
    evidence_text text,
    extraction_method text not null default 'deterministic',
    created_at timestamptz not null default now(),
    unique (source_section_id, target_section_id, relation_type)
);

create index if not exists section_edges_source_idx
    on public.section_edges(source_section_id);
create index if not exists section_edges_target_idx
    on public.section_edges(target_section_id);
create index if not exists section_edges_relation_idx
    on public.section_edges(relation_type);
create index if not exists sections_embedding_hnsw_idx
    on public.sections using hnsw (embedding extensions.vector_cosine_ops);

alter table public.section_edges enable row level security;

drop policy if exists "read edges of current documents" on public.section_edges;
create policy "read edges of current documents"
on public.section_edges for select to anon, authenticated
using (
    exists (
        select 1
        from public.sections source_section
        join public.documents document on document.id = source_section.document_id
        where source_section.id = section_edges.source_section_id
          and document.status = 'current'
    )
);

grant select on public.section_edges to anon, authenticated;

create or replace function public.expand_section_graph(
    p_seed_ids uuid[],
    p_max_depth integer default 2,
    p_limit integer default 50
)
returns table (
    section_id uuid,
    depth integer,
    graph_score double precision,
    section_path uuid[],
    relation_path text[]
)
language sql
stable
security invoker
set search_path = public
as $$
with recursive walk as (
    select
        seed.section_id,
        0 as depth,
        1.0::double precision as graph_score,
        array[seed.section_id]::uuid[] as section_path,
        array[]::text[] as relation_path
    from unnest(p_seed_ids) as seed(section_id)

    union all

    select
        edge.target_section_id,
        walk.depth + 1,
        walk.graph_score * edge.confidence *
            case edge.relation_type
                when 'exception_to' then 0.95
                when 'amends' then 0.95
                when 'supersedes' then 0.95
                when 'defines' then 0.90
                when 'references' then 0.85
                when 'requires' then 0.85
                when 'applies_to' then 0.80
                when 'translation_of' then 0.70
                when 'next_article' then 0.45
                when 'previous_article' then 0.45
                when 'same_topic' then 0.30
                else 0.25
            end,
        walk.section_path || edge.target_section_id,
        walk.relation_path || edge.relation_type
    from walk
    join public.section_edges edge on edge.source_section_id = walk.section_id
    where walk.depth < least(greatest(p_max_depth, 0), 2)
      and not edge.target_section_id = any(walk.section_path)
),
best_paths as (
    select distinct on (walk.section_id)
        walk.section_id,
        walk.depth,
        walk.graph_score,
        walk.section_path,
        walk.relation_path
    from walk
    order by walk.section_id, walk.graph_score desc
)
select *
from best_paths
order by graph_score desc
limit least(greatest(p_limit, 1), 50);
$$;

grant execute on function public.expand_section_graph(uuid[], integer, integer)
to anon, authenticated;
