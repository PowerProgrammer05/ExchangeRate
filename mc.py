import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import pandas as pd
import Datas as d

def mc(S0, mu, sig, dt, N, sim):
    S = np.zeros((sim, N))
    S[:, 0] = S0

    np.random.seed(50)
    Z = np.random.normal(size=(sim, N-1)) #Random Number for Normal Distribution

    for t in range(1, N):
        S[:, t] = S[:, t - 1] * np.exp((mu - 0.5 * sig ** 2) * dt + sig * np.sqrt(dt) * Z[:, t - 1])

    return S


if __name__ == '__main__':
    '''
    Variables
    S0 = 100 #Initial price
    mu = 0.1 #Expected Income
    sig = 0.2 #voliality (year)
    dt = 1/252 #frequency
    N = 252 #Timestamp (How long to predict)
    sim = 1000 #simulation
    '''

#line form: sig, N, Mu
#-------------------------------------------------------------INIT-------------------------------------------------------------
    sim = 1000000
#-------------------------------------------------------------KODEX------------------------------------------------------------
    plt.figure(1)
    kweights = np.ones(17)

    #KODEX LEARNING

    for i in range(1, 4):
        if i == 1:
            kodex = d.kodex_1 #sig, N, Mu
        elif i == 2:
            kodex = d.kodex_2
        else:
            kodex = d.kodex_3
        kodexval = pd.read_csv(f'/Users/krx/Documents/Codes/수연화/Kodex/Kodex_{i}_only_val.csv')
        kodex_p1 = pd.read_csv(f'/Users/krx/Documents/Codes/수연화/Kodex/Kodex_p_{i}_only_val.csv')
        kodexaly = mc(kodexval.iloc[0, 1], kodex[2], kodex[0], 1/17, 17, sim) #S0, mu, sig, dt, N, sim)
        kodexavg = np.mean(kodexaly, axis=0)

        for i in range(len(kweights)):
            kodexavg[i] *= kweights[i]

        #error
        error = 0
        errorlist = []
        maxerror = 0
        minerror = 0
        for i in range(len(kodexavg)):
            e = (kodexavg[i] - kodexval.iloc[i, 1])
            if e >= 0 and e > maxerror:
                maxerror = e
            elif e <=0 and e < minerror:
                minerror = e
            error += abs(e)
            errorlist.append(e)
        error /= 17

        for i in range(len(kweights)):
            kweights[i] = kweights[i] * (errorlist[i] / error)

    kodex4 = d.kodex_4
    rkodexval = pd.read_csv(f'/Users/krx/Documents/Codes/수연화/Kodex/Kodex_{4}_only_val.csv')
    rkodex_p1 = pd.read_csv(f'/Users/krx/Documents/Codes/수연화/Kodex/Kodex_p_{4}_only_val.csv')
    rkodexaly = mc(rkodexval.iloc[0, 1], kodex4[2], kodex4[0], 1/17, 17, sim) #S0, mu, sig, dt, N, sim)
    rkodexavg = np.mean(rkodexaly, axis=0)


    for i in range(len(kweights)):
            kodexavg[i] *= kweights[i]

    plt.suptitle('KODEX 미국반도체 MV')
    plt.subplot(2, 2, 1)
    plt.title('Actual ETF')
    plt.plot(rkodex_p1.iloc[:, 1].tolist(), linewidth=0.8, color='blue', label='Actual ETF')

    plt.subplot(2, 2, 2)
    plt.title('Prediction with weights')
    plt.plot(rkodexavg, linewidth=0.8, color='blue', label='Prediction')

    plt.subplot(2, 2, 3)
    plt.title('Montecarlo Simulation')
    for i in range(500):
        plt.plot(rkodexaly[i, :], linewidth=0.5, alpha=0.5, color='blue')

    plt.subplot(2, 2, 4)
    plt.title('Frequency of Prices')
    frequencies, bins = np.histogram(rkodexaly, bins=50)
    plt.barh(bins[:-1], frequencies, height=np.diff(bins), color='blue', alpha=0.7, label="Frequency of Prices")
    plt.ylim(min(bins), max(bins))

#-------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------TIGER=Software------------------------------------------------------------
    plt.figure(2)
    tsweights = np.ones(17)

    #Tiger-Software LEARNING

    for i in range(1, 2):
        if i == 1:
            ts = d.tiger_1 #sig, N, Mu
        elif i == 2:
            ts = d.tiger_2

        tsval = pd.read_csv(f'/Users/krx/Documents/Codes/수연화/Tiger/Tiger_{i}_only_val.csv')
        ts_p1 = pd.read_csv(f'/Users/krx/Documents/Codes/수연화/Tiger/Tiger_p_{i}_only_val.csv')
        tsaly = mc(tsval.iloc[0, 1], ts[2], ts[0], 1/17, 17, sim) #S0, mu, sig, dt, N, sim)
        tsavg = np.mean(tsaly, axis=0)

        for i in range(len(tsweights)):
            tsavg[i] *= tsweights[i]

        #error
        error = 0
        errorlist = []
        maxerror = 0
        minerror = 0
        for i in range(len(tsavg)):
            e = (tsavg[i] - tsval.iloc[i, 1])
            if e >= 0 and e > maxerror:
                maxerror = e
            elif e <=0 and e < minerror:
                minerror = e
            error += abs(e)
            errorlist.append(e)
        error /= 17

        for i in range(len(tsweights)):
            tsweights[i] = tsweights[i] * (errorlist[i] / error)

    ts3 = d.tiger_3
    rtsval = pd.read_csv(f'/Users/krx/Documents/Codes/수연화/Tiger/Tiger_{3}_only_val.csv')
    rts_p1 = pd.read_csv(f'/Users/krx/Documents/Codes/수연화/Tiger/Tiger_p_{3}_only_val.csv')
    rtsaly = mc(rtsval.iloc[0, 1], ts3[2], ts3[0], 1/17, 17, sim) #S0, mu, sig, dt, N, sim)
    rtsavg = np.mean(rtsaly, axis=0)


    for i in range(len(tsweights)):
            tsavg[i] *= tsweights[i]

    plt.suptitle('TIGER 소프트웨어')
    plt.subplot(2, 2, 1)
    plt.title('Actual ETF')
    plt.plot(rts_p1.iloc[:, 1].tolist(), linewidth=0.8, color='green', label='Actual ETF')

    plt.subplot(2, 2, 2)
    plt.title('Prediction with weights')
    plt.plot(rtsavg, linewidth=0.8, color='green', label='Prediction')

    plt.subplot(2, 2, 3)
    plt.title('Montecarlo Simulation')
    for i in range(500):
        plt.plot(rtsaly[i, :], linewidth=0.5, alpha=0.5, color='green')

    plt.subplot(2, 2, 4)
    plt.title('Frequency of Prices')
    frequencies, bins = np.histogram(rtsaly, bins=50)
    plt.barh(bins[:-1], frequencies, height=np.diff(bins), color='green', alpha=0.7, label="Frequency of Prices")
    plt.ylim(min(bins), max(bins))

#-------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------TIGER-Battery------------------------------------------------------------
    plt.figure(3)
    tbweights = np.ones(17)

    #Tiger-Software LEARNING
    for i in range(1, 2):
        if i == 1:
            tb = d.bnk_1 #sig, N, Mu
        elif i == 2:
            tb = d.bnk_2

        tbval = pd.read_csv(f'/Users/krx/Documents/Codes/수연화/Bnk/Bnk_{i}_only_val.csv')
        tb_p1 = pd.read_csv(f'/Users/krx/Documents/Codes/수연화/Bnk/Bnk_p_{i}_only_val.csv')
        tbaly = mc(tbval.iloc[0, 1], tb[2], tb[0], 1/17, 17, sim) #S0, mu, sig, dt, N, sim)
        tbavg = np.mean(tbaly, axis=0)

        for i in range(len(tbweights)):
            tbavg[i] *= tbweights[i]

        #error
        error = 0
        errorlist = []
        maxerror = 0
        minerror = 0
        for i in range(len(tbavg)):
            e = (tbavg[i] - tbval.iloc[i, 1])
            if e >= 0 and e > maxerror:
                maxerror = e
            elif e <= 0 and e < minerror:
                minerror = e
            error += abs(e)
            errorlist.append(e)
        error /= 17

        for i in range(len(tbweights)):
            tbweights[i] = tbweights[i] * (errorlist[i] / error)

    tb3 = d.bnk_3
    rtbval = pd.read_csv(f'/Users/krx/Documents/Codes/수연화/Bnk/Bnk_{3}_only_val.csv')
    rtb_p1 = pd.read_csv(f'/Users/krx/Documents/Codes/수연화/Bnk/Bnk_p_{3}_only_val.csv')
    rtbaly = mc(rtbval.iloc[0, 1], tb3[2], tb3[0], 1/17, 17, sim) #S0, mu, sig, dt, N, sim)
    rtbavg = np.mean(rtbaly, axis=0)


    for i in range(len(tbweights)):
        tbavg[i] *= tbweights[i]

    plt.suptitle('TIGER 2차전지테마')
    plt.subplot(2, 2, 1)
    plt.title('Actual ETF')
    plt.plot(rtb_p1.iloc[:, 1].tolist(), linewidth=0.8, color='orange', label='Actual ETF')

    plt.subplot(2, 2, 2)
    plt.title('Prediction with weights')
    plt.plot(rtbavg, linewidth=0.8, color='orange', label='Prediction')

    plt.subplot(2, 2, 3)
    plt.title('Montecarlo Simulation')
    for i in range(500):
        plt.plot(rtbaly[i, :], linewidth=0.5, alpha=0.5, color='orange')

    plt.subplot(2, 2, 4)
    plt.title('Frequency of Prices')
    frequencies, bins = np.histogram(rtbaly, bins=50)
    plt.barh(bins[:-1], frequencies, height=np.diff(bins), color='orange', alpha=0.7, label="Frequency of Prices")
    plt.ylim(min(bins), max(bins))

    plt.show()