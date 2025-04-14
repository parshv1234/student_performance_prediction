import numpy as np
import ast

def process_grades(grade_list):
    grade_mapping = {'S': 10, 'A': 9, 'B': 8, 'C': 7, 'D': 6, 'E': 5, 'F': 0, 'N': 0}
    grades = ast.literal_eval(grade_list) if isinstance(grade_list, str) else grade_list
    grades = [grade_mapping[g] for g in grades]
    return np.mean(grades), np.std(grades)

def process_midterms(midterm_list, target_gpa):
    scores = ast.literal_eval(midterm_list) if isinstance(midterm_list, str) else midterm_list
    avg_midterm = np.mean(scores)
    target_gpa = float(target_gpa)  # Ensure it's a scalar
    weighted_midterm = np.sum(scores) * target_gpa / len(scores)
    return avg_midterm, weighted_midterm
