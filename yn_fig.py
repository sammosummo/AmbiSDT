import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sb
sb.set_style('white')

d = 2.6
c = 0.55
k = d/2. + c
psi = np.linspace(-3, 3+d, 10000)
first = norm.pdf(psi, 0, 1)
second = norm.pdf(psi, d, 1)

fig, ((ax1, ax2), (_, _)) = plt.subplots(2, 2, sharex=False, sharey=True)
plt.xlim(-3, 3+d)
plt.ylim(0, first.max()+0.05)

ax1.plot(psi, first, label='$\Psi_{0}$', linewidth=2)
ax1.plot(psi, second, label='$\Psi_{1}$', linewidth=2)
ax1.vlines(k, 0, norm.pdf(psi, d, 1).max()*1.25, linewidth=1, zorder=10)
ax1.vlines([d/2.], 0, 0.31, linestyles='dashed', zorder=10, linewidth=1)
ax1.vlines([0, d], 0, first.max(), linestyles='dashed', zorder=10, linewidth=1)
x = np.linspace(k, 3+d, 10000)
ax1.fill_between(x, norm.pdf(x, 0, 1), norm.pdf(x, d, 1), alpha=0.3, color='g')
ax1.fill_between(x, np.zeros(x.size), norm.pdf(x, 0, 1), alpha=0.3, color='b')
ax1.spines['left'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.yaxis.set_ticks_position('none')
ax1.xaxis.set_ticks_position('bottom')
ax1.set_yticks([])
ax1.set_xticks([0, d/2., k, d])
ax1.set_xticklabels(['$0$', r'$\frac{d}{2}$', '$k$', '$d$'])
ax1.text((d/2. + k)/2. - 0.05, 0.31, '$c$')
ax1.annotate(
    "",
    xy=(d/2.-0.01, 0.3), xycoords='data',
    xytext=(k+0.01, 0.3), textcoords='data',
    arrowprops=dict(arrowstyle="->",
    connectionstyle="arc3")
    )
ax1.annotate(
    "",
    xy=(k+0.01, 0.3), xycoords='data',
    xytext=(d/2.-0.01, 0.3), textcoords='data',
    arrowprops=dict(arrowstyle="->",
    connectionstyle="arc3")
    )
ax1.text(k+0.25, first.max()+0.03, '$Y=1$')
ax1.text(k-1.2, first.max()+0.03, '$Y=0$')

c = 0.4
u = 1.2
a = d/2. + c - (u/2.)
b = d/2. + c + (u/2.)
ax2.plot(psi, first, label='$\Psi_{0}$', linewidth=2)
ax2.plot(psi, second, label='$\Psi_{1}$', linewidth=2)
ax2.legend()
ax2.vlines([a, b], 0, norm.pdf(psi, d, 1).max()*1.25, linewidth=1, zorder=10)
ax2.vlines([d/2., (a+b)/2.], 0, 0.31, linestyles='dashed', zorder=10, linewidth=1)
ax2.vlines([0, d], 0, first.max(), linestyles='dashed', zorder=10, linewidth=1)
x = np.linspace(b, 3+d, 10000)
ax2.fill_between(x, norm.pdf(x, 0, 1), norm.pdf(x, d, 1), alpha=0.3, color='g')
ax2.fill_between(x, np.zeros(x.size), norm.pdf(x, 0, 1), alpha=0.3, color='b')
y = np.linspace(a, b, 10000)
ax2.fill_between(y[:-5], np.zeros(y.size)[:-5], norm.pdf(y, 0, 1)[:-5], alpha=0.3, color='r')
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.yaxis.set_ticks_position('none')
ax2.xaxis.set_ticks_position('bottom')
ax2.set_yticks([])
ax2.set_xticks([0, d/2., (a+b)/2., d, a, b])
ax2.set_xticklabels(['$0$', r'$\frac{d}{2}$', r'$\frac{a+b}{2}$', '$d$', '$a$', '$b$'])
ax2.text((d/2. + (a+b)/2.)/2. - 0.05, 0.31, '$\lambda$')
ax2.annotate(
    "",
    xy=(d/2.-0.05, 0.3), xycoords='data',
    xytext=((a+b)/2.+0.05, 0.3), textcoords='data',
    arrowprops=dict(arrowstyle="->",
    connectionstyle="arc3")
    )
ax2.annotate(
    "",
    xy=((a+b)/2.+0.05, 0.3), xycoords='data',
    xytext=(d/2.-0.05, 0.3), textcoords='data',
    arrowprops=dict(arrowstyle="->",
    connectionstyle="arc3")
    )
ax2.text((a+b)/2. - 0.05, 0.41, '$u$')
ax2.annotate(
    "",
    xy=(a, 0.4), xycoords='data',
    xytext=(b, 0.4), textcoords='data',
    arrowprops=dict(arrowstyle="->",
    connectionstyle="arc3")
    )
ax2.annotate(
    "",
    xy=(b, 0.4), xycoords='data',
    xytext=(a, 0.4), textcoords='data',
    arrowprops=dict(arrowstyle="->",
    connectionstyle="arc3")
    )
ax2.text(b+0.15, first.max()+0.03, '$Z=1$')
ax2.text(a-1.2, first.max()+0.03, '$Z=0$')
ax2.text(d/2.-0.1, first.max()+0.03, '$Z=1$')
# # plt.gca().annotate(
# #     "",
# #     xy=(d/2.-0.5-0.5, 0.475), xycoords='data',
# #     xytext=(d/2.+0.1-0.5, 0.475), textcoords='data',
# #     arrowprops=dict(arrowstyle="->",
# #     connectionstyle="arc3")
# #     )
# # plt.text(k - 0.7, 0.45, '$Y=0$')
# # plt.legend()
# #
# # u = 1.1
# # a = d/2. + c - (u/2.)
# # b = d/2. + c + (u/2.)
# # plt.subplot(212)
# # plt.vlines([a, b], 0, norm.pdf(psi, d, 1).max()*1.25, linewidth=2, zorder=10)
# # plt.vlines([0, d/2., d/2.+c, d], 0, norm.pdf(psi, d, 1).max(), linestyles='dashed', zorder=10, linewidth=1)
# # plt.plot(psi, first, label='$\Psi_{X=0}$', linewidth=2)
# # plt.plot(psi, second, label='$\Psi_{X=1}$', linewidth=2)
# # x = np.linspace(b, 3+d, 10000)
# # plt.fill_between(x, norm.pdf(x, 0, 1), norm.pdf(x, d, 1), alpha=0.3, color='g')
# # plt.fill_between(x, np.zeros(x.size), norm.pdf(x, 0, 1), alpha=0.3, color='b')
# # y = np.linspace(a, b, 10000)
# # plt.fill_between(y, np.zeros(y.size), norm.pdf(y, 0, 1), alpha=0.3, color='r')
# # plt.gca().spines['left'].set_visible(False)
# # plt.gca().spines['right'].set_visible(False)
# # plt.gca().spines['top'].set_visible(False)
# # plt.gca().yaxis.set_ticks_position('none')
# # plt.gca().xaxis.set_ticks_position('bottom')
# # plt.gca().set_yticks([])
# # plt.gca().set_xticks([0, d/2., a, b, d, k])
# # plt.gca().set_xticklabels(['$0$', r'$\frac{d}{2}$','$a$', '$b$', '$d$', r'$\frac{a+b}{2}$'])
# # plt.gca().annotate(
# #     "",
# #     xy=((b+d)/2., 0.01), xycoords='data',
# #     xytext=(d+0.5, 0.07), textcoords='data',
# #     arrowprops=dict(arrowstyle="->",
# #     connectionstyle="arc3")
# #     )
# # plt.xlim(-3, 3+d)
# # plt.ylim(0, norm.pdf(x, d, 1).max()*1.25)
# # plt.text((d/2. + k)/2. - 0.05, 0.31, '$c$')
# # plt.gca().annotate(
# #     "",
# #     xy=(d/2., 0.3), xycoords='data',
# #     xytext=(k, 0.3), textcoords='data',
# #     arrowprops=dict(arrowstyle="->",
# #     connectionstyle="arc3")
# #     )
# # plt.gca().annotate(
# #     "",
# #     xy=(k, 0.3), xycoords='data',
# #     xytext=(d/2., 0.3), textcoords='data',
# #     arrowprops=dict(arrowstyle="->",
# #     connectionstyle="arc3")
# #     )
# # plt.text((d/2. + k)/2. - 0.05, 0.4, '$u$')
# # plt.gca().annotate(
# #     "",
# #     xy=(a, 0.4), xycoords='data',
# #     xytext=(b, 0.4), textcoords='data',
# #     arrowprops=dict(arrowstyle="->",
# #     connectionstyle="arc3")
# #     )
# # plt.gca().annotate(
# #     "",
# #     xy=(b, 0.4), xycoords='data',
# #     xytext=(a, 0.4), textcoords='data',
# #     arrowprops=dict(arrowstyle="->",
# #     connectionstyle="arc3")
# #     )
# # plt.text(k + 0.25+0.3, 0.45, '$Y=2$')
# # plt.gca().annotate(
# #     "",
# #     xy=(d+0.5+0.3, 0.475), xycoords='data',
# #     xytext=(d-0.1, 0.475), textcoords='data',
# #     arrowprops=dict(arrowstyle="->",
# #     connectionstyle="arc3")
# #     )
# # plt.gca().annotate(
# #     "",
# #     xy=(d/2.-0.5-0.5-0.5, 0.475), xycoords='data',
# #     xytext=(d/2.+0.1-0.5+0.5, 0.475), textcoords='data',
# #     arrowprops=dict(arrowstyle="->",
# #     connectionstyle="arc3")
# #     )
# # plt.text(k - 0.7-0.5, 0.45, '$Y=0$')
# # plt.text(d/2, 0.45, '$Y=1$')
plt.tight_layout(w_pad=1, h_pad=2)
plt.savefig('tmp.pdf', dpi=300)
plt.show()
