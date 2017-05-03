import numpy as np
import argparse

def build_map_3x4(initial_reward=-0.1):
    map = np.zeros((3, 4), dtype=np.float32)
    map[:] = initial_reward
    map[0, 3] = +1.0
    map[1, 3] = -1.0
    map[1, 1] = np.nan
    terms = []
    terms.append((0, 3))
    terms.append((1, 3))
    return map, terms

def init_A():
    return {
        'N': lambda s: (s[0] - 1, s[1]),
        'E': lambda s: (s[0], s[1] + 1),
        'S': lambda s: (s[0] + 1, s[1]),
        'W': lambda s: (s[0], s[1] - 1),
    }

def init_S(map, terms):
    return {
        'space': [(r, c) for r in range(map.shape[0]) for c in range(map.shape[1]) if not np.isnan(map[r, c])],
        'terms': terms[:],
    }

def init_R(map):
    return lambda s: map[s[0], s[1]]

def init_T(S, A):
    prob = {
        'N': {'N': .8, 'E': .1, 'W': .1, },
        'E': {'E': .8, 'S': .1, 'N': .1, },
        'S': {'S': .8, 'E': .1, 'W': .1, },
        'W': {'W': .8, 'S': .1, 'N': .1, },
    }
    act = lambda s, a: A[a](s) if A[a](s) in S['space'] else s
    return lambda s, a: np.array([[act(s, a_), p] for a_, p in prob[a].items()])

def init_V(S):
    return {s: .0 for s in S['space']}

def update_V(V, S, A, T, R, gamma=.9):
    def v(s):
        if s in S['terms']:
            return R(s)
        future = np.max([np.sum([V[s_] * p for s_, p in T(s, a)]) for a in A]) 
        return R(s) + gamma * future
    V_ = {s: v(s) for s in V.keys()}
    delta = np.mean([abs(V_[s] - V[s]) for s in V.keys()])
    return V_, delta

def print_V(V, S):
    Vs = [['{:+.2f}'.format(V[r, c]) if (r, c) not in terms and (r, c) in S['space'] else '-----' for c in range(map.shape[1])] for r in range(map.shape[0])]
    print('V=')
    for r in range(map.shape[0]):
        print(Vs[r])

def select_best_policy(V, S, A, T):
    def best_a(s):
        if s in S['terms']:
            return ' '
        a_ = np.argmax([np.sum([V[s_] * p for s_, p in T(s, a)]) for a in A]) 
        A_ = [a for a in A]
        return A_[a_]
    A_ = {s: best_a(s) for s in V.keys()}
    return A_

def print_policy(P, S):
    Ps = [[P[r, c] if (r, c) not in terms and (r, c) in S['space'] else ' ' for c in range(map.shape[1])] for r in range(map.shape[0])]
    print('P=')
    for r in range(map.shape[0]):
        print(Ps[r])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MDP simple")
    parser.add_argument("--initial_reward", type=float, default=-.1, help="initial reward for R(s)")
    parser.add_argument("--gamma", type=float, default=.99, help="deduction gamma for future U(s) value")
    args = parser.parse_args()
    print(args)

    map, terms = build_map_3x4(initial_reward=args.initial_reward)
    start = (2, 0)
    print('map=\n', map)
    print('terms=\n', terms)

    A = init_A()
    #print('A=', A)
    #print('A['E'](start)=', A['E'](start))
    #print('A['N'](start)=', A['N'](start))

    S = init_S(map, terms)
    #print('S=', S)

    R = init_R(map)
    #print('R=', R)
    #print('R(start)=', R(start))

    T = init_T(S, A)
    #print('T=', T)
    #print('T(start, 'E')=', T(start, 'E'))

    V = init_V(S)
    #print('V=', V)
    for iter in range(301):
        V, delta = update_V(V, S, A, T, R, gamma=args.gamma)
        if delta < 1e-5:
            print('break at iter={}'.format(iter))
            break
    #print('V=', V)

    P = select_best_policy(V, S, A, T)
    print_V(V, S)
    print_policy(P, S)
