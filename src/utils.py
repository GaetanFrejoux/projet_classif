from scipy import stats

def reduce(data):
    return stats.mode(data, keepdims=True)[0][0]