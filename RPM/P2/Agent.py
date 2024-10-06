# Allowable libraries:
# - Python 3.10.12
# - Pillow 10.0.0
# - numpy 1.25.2
# - OpenCV 4.6.0 (with opencv-contrib-python-headless 4.6.0.66)

# To activate image processing, uncomment the following imports:
# from PIL import Image
import numpy as np
import cv2

class Agent:
    def __init__(self):
        """
        The default constructor for your Agent. Make sure to execute any processing necessary before your Agent starts
        solving problems here. Do not add any variables to this signature; they will not be used by main().
        
        This init method is only called once when the Agent is instantiated 
        while the Solve method will be called multiple times. 
        """
        pass

    def Solve(self, problem):
        """
        Primary method for solving incoming Raven's Progressive Matrices.

        Args:
            problem: The RavensProblem instance.

        Returns:
            int: The answer (1-6 for 2x2 OR 1-8 for 3x3).
            Return a negative number to skip a problem.
            Remember to return the answer [Key], not the name, as the ANSWERS ARE SHUFFLED in Gradescope.
        """

        '''
        DO NOT use absolute file pathing to open files.
        
        Example: Read the 'A' figure from the problem using Pillow
            image_a = Image.open(problem.figures["A"].visualFilename)
            
        Example: Read the '1' figure from the problem using OpenCv
            image_1 = cv2.imread(problem.figures["1"].visualFilename)
            
        Don't forget to uncomment the import!
        '''

        answer = get_answer(problem, problem.problemType)
        return answer
    
trans_dcit = {
    'rotated_90': lambda image: cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
    'rotated_180': lambda image: cv2.rotate(image, cv2.ROTATE_180),
    'rotated_270': lambda image: cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
    'flipped_vertical': lambda image: cv2.flip(image, 0),
    'flipped_horizontal': lambda image: cv2.flip(image, 1)
}

def similarity(image1, image2):
# Calculate the normalized cross-correlation (NCC)
    ncc = cv2.matchTemplate(image1, image2, cv2.TM_CCOEFF_NORMED)
    return ncc

def compare(image_pair, image_new):
    
    image_original, image_transfromed = image_pair
    trans_list = []
    for trans, trans_func in trans_dcit.items():
        if similarity(trans_func(image_original), image_transfromed) > 0.85:
            trans_list.append(trans)
            # print(trans)
            # show(trans_dcit[trans](image_new))
    return [trans_dcit[trans](image_new) for trans in trans_list]

def tackle_2x2(problem):
    
    answers = [cv2.imread(problem.figures[str(index)].visualFilename) for index in range(1, 7)]
    figure_a = cv2.imread(problem.figures['A'].visualFilename)
    figure_b = cv2.imread(problem.figures['B'].visualFilename)
    figure_c = cv2.imread(problem.figures['C'].visualFilename)

    candidate_list = compare([figure_a, figure_b], figure_c) + compare([figure_a, figure_c], figure_b)
    candidate_list += [figure_c + figure_b - figure_a]

    # return candidate_list
    match_count_list = []
    for answer in answers:
        match_count_list.append(max([similarity(answer, candidate) for candidate in candidate_list]))
        # match_count_list.append(sum([1 if similarity(answer, candidate) > 0.95 else 0 for candidate in candidate_list]))

    # return match_count_list
    max_count = max(match_count_list)
    # Get the index of the maximum element
    my_answer = match_count_list.index(max_count)

    return my_answer + 1

def get_answer(problem, problem_type):
    answer = -1
    
    if problem_type == "2x2":
        answer = tackle_2x2(problem)
    
    return answer




