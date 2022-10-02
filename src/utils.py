from scipy import stats

def stats_reduce(data):
    return stats.mode(data, keepdims=True)[0][0]