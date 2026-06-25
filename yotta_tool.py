import asyncio
import requests
import traceback
import json


yotta_api = "http://216.151.16.11:5155/api/question/answer-engine"
api_key = "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoiZGV2X3VzZXIiLCJleHAiOjMyNTAzNjgwMDAwfQ.zZcSMdiMSSuT7p5IQG-l9IUbR3A9aUlzYkESiObCgDM"

def process_serper_kg_entites(kg_data):
    kg_list = []
    if kg_data is None:
        return []

    entity = kg_data['title']
    if 'attributes' in kg_data:
        for k1,v1 in kg_data['attributes'].items():
            kg_list.append((entity, k1, v1))

    if 'type' in kg_data:
        kg_list.append((entity, 'is', kg_data['type']))

    return [f"({r[0]},{r[1]},{r[2]})"for r in kg_list]

def _call_yotta_sync(question: str, answers_returned: int=10, sents_per_answer: int=1) -> str:
    """Synchronous core: see ``call_yotta`` for the public async API."""

    body = {"question":question,"version":"v3","sents_per_answer":sents_per_answer}
    headers = {"Content-Type": "application/json", "Authorization": api_key}


    try:
        response = requests.post(yotta_api,json=body,headers=headers)
        answers = response.json()

        answer_list = [{'answer':answer['answer'],'sentence':answer['sentence'],'source':answer['sources'][0]['url']} for answer in answers['answers']]
        kg_data = answers.get("knowledge_graph",None)

        kg_website = ""
        if kg_data:
            if 'descriptionLink' in kg_data:
                kg_website = kg_data['descriptionLink']
            elif 'website' in kg_data:
                kg_website = kg_data['website']
            kg_triplets = process_serper_kg_entites(kg_data)
            kg_data = '\n'.join(kg_triplets)
        else:
            kg_data = ""

        sentences = ["{0}{{{1}}}".format(answer['sentence'],answer['source']) for answer in answer_list][:answers_returned]
        kg_output = "{0}{{{1}}}".format(kg_data, kg_website)
        if len(sentences) == 0:
            sentences = ["No good answer found."]
        sources = '.\n'.join(sentences)

        return f"<source>\n{sources}</source>\n<graph>{kg_output}</graph>"
    except:
        print(traceback.format_exc())
        print("ERROR: ",question,response)
        return ""


def parse_yotta_results(yotta_raw: str) -> str:
    """
    Convert raw yotta XML output into clean markdown suitable for the
    orchestrator and writer nodes.

    Input format (from ``_call_yotta_sync``):
        <source>
        sentence1{{url1}}.
        sentence2{{url2}}
        </source>
        <graph>(entity,rel,val)\\n(entity,is,type){website}</graph>

    Output format:
        - sentence1 [source: url1]
        - sentence2 [source: url2]
        ...
        ## Knowledge graph
        - (entity, rel, val)
        ...
    """
    import re

    blocks: list[str] = []

    # --- <source> block -------------------------------------------------------
    src_match = re.search(r"<source>\s*(.*?)\s*</source>", yotta_raw, re.DOTALL)
    if src_match:
        sources_text = src_match.group(1)
        for line in sources_text.split("\n"):
            line = line.strip().rstrip(".")
            if not line:
                continue
            # "sentence text{{url}}"
            sent_match = re.match(r"^(.*?)\{\{(.*?)\}\}$", line)
            if sent_match:
                sentence = sent_match.group(1).strip()
                url = sent_match.group(2).strip()
                blocks.append(f"- {sentence} [source: {url}]")
            else:
                blocks.append(f"- {line}")

    # --- <graph> block --------------------------------------------------------
    graph_match = re.search(r"<graph>\s*(.*?)\s*</graph>", yotta_raw, re.DOTALL)
    if graph_match:
        kg_raw = graph_match.group(1)
        # Strip trailing {website_url}
        kg_clean = re.sub(r"\{.*?\}$", "", kg_raw).strip()
        if kg_clean:
            kg_lines = [
                f"- {ln.strip()}" for ln in kg_clean.split("\n") if ln.strip()
            ]
            if kg_lines:
                blocks.append("## Knowledge graph")
                blocks.extend(kg_lines)

    return "\n".join(blocks) if blocks else yotta_raw


async def call_yotta(question: str, answers_returned: int=10, sents_per_answer: int=1) -> str:
    """Get paragraph answer for a question. There is no Knowledge date cuttoff, as it is connected to the internet. Answering system is great for simple (one-hop) factual questions (e.g. 'What is the difference between Apple and google' is a bad question, better is to ask seperatly 'What is Apple?' and 'What is google?' and then reason over the question; 'What is history of Machine learning and its applications?' is also one of bad examples, it would be better to ask 'What is history of Machine learning' and 'What are applications of Machine learning? and then combine those answers), because it has a lot of information, but has no reasoning capability. Use it for retrieving information that are not changing (e.g. 'how old is somone?' is a bad question, better is 'when was someone born?') System uses English as primary language.
    Args:
        question: The simple factual question that needs to be answered (in English)
        answers_returned: Max number of answers returned, bigger number is better for question with precies answers
        sents_per_answer: number of sentences retrieved per answer, bigger number is better for question with non precies answers
    Returns:
        the answer, list of sentences that answers the question formated as a paragraph and knowledge graph triplets. Sentences can be inconsistencies, information should be used by position of the sentence. Knowledge graph can be empty.
    """
    return await asyncio.to_thread(
        _call_yotta_sync, question, answers_returned, sents_per_answer
    )