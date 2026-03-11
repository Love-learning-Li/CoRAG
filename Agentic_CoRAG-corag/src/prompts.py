from typing import List, Dict, Optional


def get_generate_subquery_prompt(query: str, past_subqueries: List[str], past_subanswers: List[str], task_desc: str) -> List[Dict]:
    assert len(past_subqueries) == len(past_subanswers)
    past = ''
    for idx in range(len(past_subqueries)):
        past += f"""Intermediate query {idx+1}: {past_subqueries[idx]}
Intermediate answer {idx + 1}: {past_subanswers[idx]}\n"""
    past = past.strip()

    prompt = f"""You are using a search engine to answer the main query by iteratively searching the web. Given the following intermediate queries and answers, generate exactly one new simple follow-up question that helps answer the main query.

Rules:
- Ask for exactly one missing fact in one hop.
- Preserve the relation direction from the main query. Do not swap father with grandfather, spouse with parent, or person with parent-in-law.
- Preserve the target attribute exactly. For example, do not replace nationality with workplace, country with headquarters city, or mother with maternal grandmother.
- For kinship questions, first resolve the directly needed relative, then ask for the next relation on that resolved person.
- For comparison questions (e.g., "Are both X and Y from the same country?"), ask for the SAME canonical attribute (e.g., country name) for each entity separately. When asking about country, ask at the country level—not city, state, or province.
- For "[attribute] of [role] of [entity]" questions (e.g., "What nationality is the director of film X?", "Where did the composer of song Y die?"), FIRST resolve who the role is (ask "Who is the director of film X?"), THEN ask for the attribute of the resolved person. Do NOT try to answer the attribute directly in the first step.
- If a previous intermediate answer is "No relevant information found", rephrase the SAME question using alternative keywords or a more specific variant (e.g., use past tense, add context clues). Do NOT ask a completely different or unrelated question.
- Keep the query short and retrieval-friendly.
- Do not explain yourself and do not output multiple questions.
- If you already have all the intermediate answers needed to answer the main query (i.e., no more facts are missing), output exactly the text: [STOP]

## Previous intermediate queries and answers
{past or 'Nothing yet'}

## Task description
{task_desc}

## Main query to answer
{query}

Respond with one simple follow-up question only."""

    messages: List[Dict] = [
        {'role': 'user', 'content': prompt}
    ]
    return messages


def get_generate_intermediate_answer_prompt(subquery: str, documents: List[str]) -> List[Dict]:
    context = ''
    for idx, doc in enumerate(documents):
        context += f"""{doc}\n\n"""

    prompt = f"""Given the following documents, generate an appropriate answer for the query.

Rules:
- DO NOT hallucinate. Only use the provided documents.
- If the query asks for a single person, place, date, country, employer, or award, answer with the single best-supported value only.
- Do not list multiple candidates unless the query explicitly asks for multiple answers.
- Prefer the canonical short answer span from the documents.
- Respond "No relevant information found" if the documents do not contain useful information.
- For role-holder queries (e.g., "Who is the composer/director/performer/author/writer of X?"), actively look for patterns like "composed by Y", "directed by Y", "performed by Y", "written by Y", "music by Y", "starring Y". Extract the name Y directly from such phrases.
- For family-member queries (e.g., "Who is X's mother/father/spouse?"), scan for "daughter/son of X", "wife/husband of X", "married to X", "his wife Y", "her husband Y", "X's [relation] was Y", "Y was the [relation] of X".
- If asked for a COUNTRY, return the country name only—not a city, state, or province. If the document mentions a US state (California, Texas, New York, etc.) or US territory (Puerto Rico, Guam, etc.), the country answer is "United States".
- Return names WITHOUT honorific titles. Do NOT include prefixes such as Queen, King, Emperor, Empress, Prince, Princess, Grand Duke, Grand Duchess, Duke, Duchess, Tsar, Tsarina, Sir, Dame, Dr., Prof. For example, return "Marie Leszczyńska" not "Queen Marie Leszczyńska".

## Documents
{context.strip()}

## Query
{subquery}

Respond with a concise answer only, do not explain yourself or output anything else."""

    messages: List[Dict] = [
        {'role': 'user', 'content': prompt}
    ]
    return messages


def get_generate_final_answer_prompt(
        query: str, past_subqueries: List[str], past_subanswers: List[str], task_desc: str,
        documents: Optional[List[str]] = None
) -> List[Dict]:

    assert len(past_subqueries) == len(past_subanswers)
    past = ''
    for idx in range(len(past_subqueries)):
        past += f"""Intermediate query {idx+1}: {past_subqueries[idx]}
Intermediate answer {idx+1}: {past_subanswers[idx]}\n"""
    past = past.strip()

    context = ''
    if documents:
        for idx, doc in enumerate(documents):
            context += f"""Doc {idx}: {doc}\n\n"""

    prompt = f"""Given the following intermediate queries and answers, generate a final answer for the main query by combining relevant information. Note that intermediate answers are generated by an LLM and may not always be accurate.

## Documents
{context.strip()}

## Intermediate queries and answers
{past or 'Nothing yet'}

## Task description
{task_desc}

## Main query
{query}

Additional answer format rules:
- If the main query is a yes/no question (e.g., "Are both…?", "Do both…?", "Did both…?", "Were both…?", "Is X…?", "Was X…?", or contains phrases like "from the same", "in the same", "the same country/place/nationality"), respond with ONLY the single word "yes" or "no" (lowercase), nothing else.
- For geographic comparisons: US states (California, New York, Texas, Florida, etc.) and US territories (Puerto Rico, Guam, etc.) are all part of the United States. A work described as "American-Argentine" is partly from the United States. Two entities both connected to the United States share the same country.
- Return names in their shortest canonical form WITHOUT honorific titles (Queen, King, Emperor, Empress, Prince, Princess, Grand Duke, Grand Duchess, Duke, Duchess, Earl, Baron, Tsar, Tsarina, Sir, Dame, etc.). For example use "Marie Leszczyńska" not "Queen Marie Leszczyńska".
- For award names, prefer the canonical form found in the retrieved documents (original-language names are acceptable).

Respond with an appropriate answer only, do not explain yourself or output anything else."""

    messages: List[Dict] = [
        {'role': 'user', 'content': prompt}
    ]
    return messages
