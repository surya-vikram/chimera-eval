"""Conservative refusal-target/meta-prompt screening, not semantic certification."""
import json
import re

REFUSAL = re.compile(r"\b(?:I|we)\s+(?:cannot|can't|won't|will not|am unable to|are unable to)\s+(?:assist|help|provide|comply|answer)\b", re.I)
META = re.compile(r'\b(?:do anything now|do nothing now|jailbreak|as DAN|be a DNN)\b', re.I)
INSTRUCT = re.compile(r"\b(?:you|the assistant|the model|chatgpt)\s+(?:(?:must|should|will)\s+not|cannot|can't)\s+(?:answer|respond|comply)\b", re.I)


def rejection_reason(row):
    # Refusal language in a supplied evidence passage is not necessarily a refusal target.
    # Never conflate a factual story about someone unable to answer with assistant behavior.
    verifier = row.get('verifier')
    meta = row.get('verification', {})
    references = [meta.get('answer',''), meta.get('reference','')]
    for reference in references:
        if isinstance(reference,str) and REFUSAL.search(reference):
            return 'refusal_reference_target'
    for message in row.get('messages',[]):
        if message['role'] == 'assistant' and REFUSAL.search(message['content']):
            return 'refusal_in_conversation_history'
    if verifier in ('quality','rubric','instruction','structure','calendar'):
        text = '\n'.join(m['content'] for m in row.get('messages',[]))
        if META.search(text): return 'adversarial_meta_prompt'
        if INSTRUCT.search(text): return 'explicit_nonanswer_instruction'
        if REFUSAL.search(text): return 'quoted_refusal_example_requires_review'
    return None
