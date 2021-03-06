import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sb
sb.set_style('white')

d = 2.4
c = 0.4
psi = np.linspace(-3, 3+d, 10000)
first = norm.pdf(psi, 0, 1)
second = norm.pdf(psi, d, 1)
x = 0.9
k = x + c


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]


plt.subplot(211)
plt.plot([x], norm.pdf(x, d, 1), 'ko')
plt.vlines(k, 0, first.max()*1.25, linewidth=2, zorder=10)
plt.vlines(x, 0, norm.pdf(x, d, 1), linestyles='dashed', zorder=10, linewidth=1)
plt.vlines([0, d], 0, first.max(), linestyles='dashed', zorder=10, linewidth=1)
plt.plot(psi, second, label='$\Psi_{0}$', linewidth=2)
plt.plot(psi, first, label='$\Psi_{1}$', linewidth=2)
z = np.linspace(k, 3+d, 10000)
# plt.fill_between(z, norm.pdf(z, 0, 1), norm.pdf(z, d, 1), alpha=0.3, color='g')
plt.fill_between(z, np.zeros(z.size), norm.pdf(z, 0, 1), alpha=0.3, color='b')
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('none')
plt.gca().xaxis.set_ticks_position('bottom')
plt.gca().set_yticks([])
plt.gca().set_xticks([0, x, d])
plt.gca().set_xticklabels(['$0$', '$x$', '$d$'])
plt.xlim(-3, 3+d)
plt.ylim(0, norm.pdf(z, d, 1).max()*1.25)
plt.text((x + k)/2. - 0.05, 0.1, '$l$')
plt.text(x-0.35, 0.17, r'$\Psi_0=x$')
plt.gca().annotate(
    "",
    xy=(x, 0.09), xycoords='data',
    xytext=(k, 0.09), textcoords='data',
    arrowprops=dict(arrowstyle="->",
    connectionstyle="arc3")
    )
plt.gca().annotate(
    "",
    xy=(k, 0.09), xycoords='data',
    xytext=(x, 0.09), textcoords='data',
    arrowprops=dict(arrowstyle="->",
    connectionstyle="arc3")
    )
plt.text(k + 0.25, 0.45, '$Y=1$')
# plt.gca().annotate(
#     "",
#     xy=(d+0.5, 0.475), xycoords='data',
#     xytext=(d-0.1, 0.475), textcoords='data',
#     arrowprops=dict(arrowstyle="->",
#     connectionstyle="arc3")
#     )
# plt.gca().annotate(
#     "",
#     xy=(d/2.-0.5-0.5, 0.475), xycoords='data',
#     xytext=(d/2.+0.1-0.5, 0.475), textcoords='data',
#     arrowprops=dict(arrowstyle="->",
#     connectionstyle="arc3")
#     )
plt.text(k - 0.7, 0.45, '$Y=0$')
plt.legend()

u = 1.1
a = x + c - (u/2.)
b = x + c + (u/2.)
plt.subplot(212)
plt.plot([x], norm.pdf(x, d, 1), 'ko')
plt.vlines([a,b], 0, first.max()*1.25, linewidth=2, zorder=10)
plt.vlines([x], 0, norm.pdf(x, d, 1), linestyles='dashed', zorder=10, linewidth=1)
plt.vlines([k], 0, 0.1, linestyles='dashed', zorder=10, linewidth=1)
plt.vlines([0, d], 0, first.max(), linestyles='dashed', zorder=10, linewidth=1)
plt.plot(psi, second, label='$\Psi_{0}$', linewidth=2)
plt.plot(psi, first, label='$\Psi_{1}$', linewidth=2)

z = np.linspace(b, 3+d, 10000)
plt.fill_between(z, np.zeros(z.size), norm.pdf(z, 0, 1), alpha=0.3, color='b')
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('none')
plt.gca().xaxis.set_ticks_position('bottom')
plt.gca().set_yticks([])
plt.gca().set_xticks([0, x, (a+b)/2., d, a, b])
plt.gca().set_xticklabels(['$0$', '$x$', r'$\frac{a+b}{2}$', '$d$', '$a$', '$b$'])
plt.xlim(-3, 3+d)
plt.ylim(0, norm.pdf(z, d, 1).max()*1.25)
# plt.text((x + k)/2. - 0.05, 0.1, '$l$')
# plt.text(x-0.8, 0.17, r'$\Psi_0=x$')
plt.gca().annotate(
    r'$\Psi_0=x$',
    xy=(x, norm.pdf(x,d,1)), xycoords='data',
    xytext=(x-0.8, 0.26), textcoords='data',
    arrowprops=dict(arrowstyle="->",
    connectionstyle="arc3")
    )
plt.gca().annotate(
    "",
    xy=(x, 0.09), xycoords='data',
    xytext=(k, 0.09), textcoords='data',
    arrowprops=dict(arrowstyle="->",
    connectionstyle="arc3")
    )
plt.gca().annotate(
    "",
    xy=(k, 0.09), xycoords='data',
    xytext=(x, 0.09), textcoords='data',
    arrowprops=dict(arrowstyle="->",
    connectionstyle="arc3")
    )
plt.text(k + 0.6, 0.45, '$Z=2$')
#
# u = 1.1
# a = d/2. + c - (u/2.)
# b = d/2. + c + (u/2.)
# plt.subplot(212)
# plt.vlines([a, b], 0, norm.pdf(psi, d, 1).max()*1.25, linewidth=2, zorder=10)
# plt.vlines([0, d/2., d/2.+c, d], 0, norm.pdf(psi, d, 1).max(), linestyles='dashed', zorder=10, linewidth=1)
# plt.plot(psi, first, label='$\Psi_{X=0}$', linewidth=2)
# plt.plot(psi, second, label='$\Psi_{X=1}$', linewidth=2)
# x = np.linspace(b, 3+d, 10000)
# plt.fill_between(x, norm.pdf(x, 0, 1), norm.pdf(x, d, 1), alpha=0.3, color='g')
# plt.fill_between(x, np.zeros(x.size), norm.pdf(x, 0, 1), alpha=0.3, color='b')
y = np.linspace(a, b, 10000)
plt.fill_between(y, np.zeros(y.size), norm.pdf(y, 0, 1), alpha=0.3, color='r')
# plt.gca().spines['left'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().yaxis.set_ticks_position('none')
# plt.gca().xaxis.set_ticks_position('bottom')
# plt.gca().set_yticks([])
# plt.gca().set_xticks([0, d/2., a, b, d, k])
# plt.gca().set_xticklabels(['$0$', r'$\frac{d}{2}$','$a$', '$b$', '$d$', r'$\frac{a+b}{2}$'])
plt.gca().annotate(
    "$f$",
    xy=((b+d)/2., 0.01), xycoords='data',
    xytext=(d+0.5, 0.07), textcoords='data',
    arrowprops=dict(arrowstyle="->",
    connectionstyle="arc3")
    )
# plt.xlim(-3, 3+d)
# plt.ylim(0, norm.pdf(x, d, 1).max()*1.25)
# plt.text((d/2. + k)/2. - 0.05, 0.31, '$c$')
# plt.gca().annotate(
#     "",
#     xy=(d/2., 0.3), xycoords='data',
#     xytext=(k, 0.3), textcoords='data',
#     arrowprops=dict(arrowstyle="->",
#     connectionstyle="arc3")
#     )
# plt.gca().annotate(
#     "",
#     xy=(k, 0.3), xycoords='data',
#     xytext=(d/2., 0.3), textcoords='data',
#     arrowprops=dict(arrowstyle="->",
#     connectionstyle="arc3")
#     )
# plt.text((d/2. + k)/2. - 0.05, 0.4, '$u$')
# plt.gca().annotate(
#     "",
#     xy=(a, 0.4), xycoords='data',
#     xytext=(b, 0.4), textcoords='data',
#     arrowprops=dict(arrowstyle="->",
#     connectionstyle="arc3")
#     )
# plt.gca().annotate(
#     "",
#     xy=(b, 0.4), xycoords='data',
#     xytext=(a, 0.4), textcoords='data',
#     arrowprops=dict(arrowstyle="->",
#     connectionstyle="arc3")
#     )
# plt.text(k + 0.25+0.3, 0.45, '$Y=2$')
# plt.gca().annotate(
#     "",
#     xy=(d+0.5+0.3, 0.475), xycoords='data',
#     xytext=(d-0.1, 0.475), textcoords='data',
#     arrowprops=dict(arrowstyle="->",
#     connectionstyle="arc3")
#     )
# plt.gca().annotate(
#     "",
#     xy=(d/2.-0.5-0.5-0.5, 0.475), xycoords='data',
#     xytext=(d/2.+0.1-0.5+0.5, 0.475), textcoords='data',
#     arrowprops=dict(arrowstyle="->",
#     connectionstyle="arc3")
#     )
plt.text(k - 0.7-0.5, 0.45, '$Z=0$')
plt.text(d/2, 0.45, '$Z=1$')
plt.tight_layout(w_pad=0.1, h_pad=0.1)
plt.savefig('tmp.pdf', dpi=300)


