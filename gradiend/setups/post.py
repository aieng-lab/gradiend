from gradiend.setups import Setup

if __name__ == '__main__':
    setup = Setup('test')
    setup._post_training('results/experiments/gradiend/emotion-10/distilbert-base-cased/tanh_10_v1/0')
    #setup._post_training('results/experiments/gradiend/emotion-2/distilbert-base-cased/tanh_2_v2/0')