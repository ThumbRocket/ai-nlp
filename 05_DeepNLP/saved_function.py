def grade_essay(essay):
    score = 0
    if "규칙" in essay:
        score += 5
    if "템플릿" in essay or "template" in essay:
        score += 5
    if "머신러닝" in essay or "머신 러닝":
        score += 6
    if "유연성" in essay:
        score += 2
    if "시간" in essay:
        score += 2
    return score
