import random
import numpy as np
import matplotlib.pyplot as plt

N = 100  # Length of signal
K = 32  # Length of vosine matrix


def cosine(n, k):
    """Function: Cos((pi*k/2N) * (2N+1)); Sum for n=0 to N"""
    return np.cos((2 * k + 1) * (n * np.pi / (2 * N)))


def DCT(x):
    """Find all composite cosine frequencies via matrix multiplication"""
    C = np.fromfunction(lambda i, j: cosine(i, j), (K, N), dtype=float)
    out = np.matmul(x, C.T)
    return out


def main():
    # Pure cos signal for testing
    period = 4
    x_1 = np.cos(np.linspace(-np.pi, np.pi, N) * period / 2)
    x_1 += np.cos(np.linspace(-np.pi, np.pi, N) * 8 / 2)
    out1 = DCT(x_1)

    # Random signal generated for decomposition
    periods = random.sample(range(0, 16), 2)
    x_2 = np.cos(np.linspace(-np.pi, np.pi, N) * periods[0] / 2)
    x_2 += np.cos(np.linspace(-np.pi, np.pi, N) * periods[1] / 2)
    out2 = DCT(x_2)

    # Plotting
    fig, axs = plt.subplots(2, 2)
    fig.suptitle("Discrete Cosine Transform")
    fig1_domain = np.linspace(0, np.pi * 5, N)
    fig2_domain = np.arange(K)
    axs[0, 0].set_title("Sum of cos(4x) and cos(8x)")
    axs[0, 1].set_title("Sum of cos({}x) and cos({}x)".format(periods[0], periods[1]))
    axs[1, 0].set_title("")
    axs[1, 1].set_title("")
    axs[0, 0].plot(fig1_domain, x_1)
    axs[0, 1].plot(fig1_domain, x_2)
    axs[1, 0].plot(fig2_domain, out1)
    axs[1, 1].plot(fig2_domain, out2)
    plt.show()


if __name__=="__main__":
    main()
