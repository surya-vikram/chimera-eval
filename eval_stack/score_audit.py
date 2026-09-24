"""Fixed scorer unit cases, optionally through the actual hosted judge; NOT benchmark data."""
import argparse
import json
from .client import Client
from .graders import Grader
from .common import write_json


def audit(url, model, output, judge_context=8192, judge_max_tokens=512, chat_template_kwargs=None):
    kwargs = {'enable_thinking':False} if chat_template_kwargs is None else chat_template_kwargs
    client = Client(url, model, {'temperature':0, 'chat_template_kwargs':kwargs}, concurrency=1)
    grader = Grader(client, judge_context=judge_context, judge_max_tokens=judge_max_tokens)
    cases = [
        ('qa_concise_correct','alias', {'aliases':['Paris']}, 'What is the capital of France?', 'Paris', 1),
        ('qa_verbose_correct','alias', {'aliases':['Paris']}, 'What is the capital of France?', 'The capital of France is Paris. It is also its largest city and stands on the River Seine.',1),
        ('qa_concise_wrong','alias', {'aliases':['Paris']}, 'What is the capital of France?', 'Berlin',0),
        ('qa_mention_but_wrong','alias', {'aliases':['Paris']}, 'What is the capital of France?', 'Although some people say Paris, the capital of France is actually Berlin.',0),
        ('qa_multiple_conflicting','alias', {'aliases':['Paris']}, 'What is the capital of France?', 'The answer is Paris or Berlin; both are equally correct.',0),
        ('math_verbose_correct','math', {'answer':'4'}, 'What is two plus two?', 'Adding two and two gives four. This follows directly from counting two pairs.\nFinal answer: \\boxed{4}',1),
        ('math_wrong','math', {'answer':'4'}, 'What is two plus two?', 'Final answer: \\boxed{5}',0),
        ('equivalence_verbose','equivalence', {'answer':'water'}, 'What compound is H2O?', 'H2O is water, consisting of two hydrogen atoms and one oxygen atom.',1),
        ('grounding_correct','grounded', {'aliases':['Paris']}, 'Passage: Paris is the capital of France. Question: What is the capital?', 'The supplied passage says the capital of France is Paris.',1),
        ('grounding_contradiction','grounded', {'aliases':['Paris']}, 'Passage: Paris is the capital of France. Question: What is the capital?', 'Paris is in Germany.\nFinal answer: Paris',0),
        ('choice_reasoning','choice', {'answer':'B','labels':['A','B']}, '2+2? A:3 B:4', 'Adding gives four, corresponding to the second option.\nFinal answer: B',1),
        ('retrieval_complete','retrieval', {'targets':['123','456']}, 'Find both codes.', 'The requested codes are 123 and 456.',1),
        ('retrieval_incomplete','retrieval', {'targets':['123','456']}, 'Find both codes.', '123',0),
    ]
    results = []
    for name,kind,meta,prompt,answer,expected in cases:
        row = {'verifier':kind,'verification':meta,'messages':[{'role':'user','content':prompt}], 'binary':True,'domain':'knowledge'}
        try:
            grade = grader.grade(row, {'text':answer,'finish_reason':'stop'})
            results.append({'case':name,'expected':expected,'actual':grade,'matches':grade['score']==expected})
        except Exception as e:
            results.append({'case':name,'expected':expected,'error':str(e),'matches':False})
        print(json.dumps(results[-1]), flush=True)
    report = {'cases':results,'passed':sum(r['matches'] for r in results),'total':len(results),
              'purpose':'scorer plumbing / limited judge sanity, NOT judge quality certification'}
    write_json(output,report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--url',required=True); p.add_argument('--model',required=True); p.add_argument('--output',required=True)
    p.add_argument('--judge-context', type=int, default=8192)
    p.add_argument('--judge-max-tokens', type=int, default=512)
    p.add_argument('--chat-template-kwargs', type=json.loads, default=None)
    args = p.parse_args()
    audit(args.url,args.model,args.output,args.judge_context,args.judge_max_tokens,args.chat_template_kwargs)
