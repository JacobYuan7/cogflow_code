
import re

_SOLUTION_CLIP_CHARS = 300


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]


    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # this also tests the formatting of the model
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def compute_score(solution_str, ground_truth):

    # answer = extract_solution(solution_str=solution_str, method=method)
    # if answer is None:
    #     return 0
    # else:
    #     if answer == ground_truth:
    #         return score
    #     else:
    #         return format_score


    if 'Response' not in solution_str:
        pred = solution_str
    else:
        pred = solution_str.split("Response:")[-1].strip()
    gt = ground_truth['target'].split("Response:")[-1].strip()

    # 进行前缀匹配
    pred_list = pred.strip('<').strip('>').split("><")
    gt_list = gt.strip('<').strip('>').split("><")

    match_n = 0
    for pred_item in pred_list:
        if pred_item in gt_list:
            match_n += 1

    return match_n / len(gt_list)