try:
    from .cpp.evaluate import eval_rating_matrix
    print("Evaluate result with cpp.")
except:
    from .python.evaluate import eval_rating_matrix
    print("Evaluate result with python.")
