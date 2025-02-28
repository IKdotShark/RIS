import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Параметры распределений для 10 варианта
a1, b1 = 100, 5000  # Равномерное распределение U(100, 5000)
mu2, sigma2 = 500, np.sqrt(10000)  # Нормальное распределение N(500, 10000)
k3, theta3 = 8, 65  # Гамма-распределение Γ(8, 65)

# 1. Вероятность безотказной работы (Функция надежности)
def reliability_uniform(t, a, b):
    return 1 - stats.uniform.cdf(t, loc=a, scale=b-a)

def reliability_normal(t, mu, sigma):
    return 1 - stats.norm.cdf(t, loc=mu, scale=sigma)

def reliability_gamma(t, k, theta):
    return 1 - stats.gamma.cdf(t, a=k, scale=theta)

# 2. Средняя наработка до отказа (Математическое ожидание)
mean_uniform = (a1 + b1) / 2
mean_normal = mu2
mean_gamma = k3 * theta3

# 3. Дисперсия и среднее квадратическое отклонение
var_uniform = (b1 - a1)**2 / 12
std_uniform = np.sqrt(var_uniform)

var_normal = sigma2**2
std_normal = sigma2

var_gamma = k3 * theta3**2
std_gamma = np.sqrt(var_gamma)

# 4. Интенсивность отказов
def failure_rate_uniform(t, a, b):
    return stats.uniform.pdf(t, loc=a, scale=b-a) / reliability_uniform(t, a, b)

def failure_rate_normal(t, mu, sigma):
    return stats.norm.pdf(t, loc=mu, scale=sigma) / reliability_normal(t, mu, sigma)

def failure_rate_gamma(t, k, theta):
    return stats.gamma.pdf(t, a=k, scale=theta) / reliability_gamma(t, k, theta)

# 5. Плотность распределения времени до отказа
def pdf_uniform(t, a, b):
    return stats.uniform.pdf(t, loc=a, scale=b-a)

def pdf_normal(t, mu, sigma):
    return stats.norm.pdf(t, loc=mu, scale=sigma)

def pdf_gamma(t, k, theta):
    return stats.gamma.pdf(t, a=k, scale=theta)

# 6. Гамма-процентная наработка до отказа
def gamma_percentile_uniform(gamma, a, b):
    return stats.uniform.ppf(gamma/100, loc=a, scale=b-a)

def gamma_percentile_normal(gamma, mu, sigma):
    return stats.norm.ppf(gamma/100, loc=mu, scale=sigma)

def gamma_percentile_gamma(gamma, k, theta):
    return stats.gamma.ppf(gamma/100, a=k, scale=theta)

# Графики
t = np.linspace(0, 6000, 1000)

# Вероятность безотказной работы
plt.figure(figsize=(10, 6))
plt.plot(t, reliability_uniform(t, a1, b1), label='Uniform U(100, 5000)')
plt.plot(t, reliability_normal(t, mu2, sigma2), label='Normal N(500, 10000)')
plt.plot(t, reliability_gamma(t, k3, theta3), label='Gamma Γ(8, 65)')
plt.title('Вероятность безотказной работы')
plt.xlabel('Время')
plt.ylabel('Вероятность')
plt.legend()
plt.grid()
plt.show()

# Интенсивность отказов
plt.figure(figsize=(10, 6))
plt.plot(t, failure_rate_uniform(t, a1, b1), label='Uniform U(100, 5000)')
plt.plot(t, failure_rate_normal(t, mu2, sigma2), label='Normal N(500, 10000)')
plt.plot(t, failure_rate_gamma(t, k3, theta3), label='Gamma Γ(8, 65)')
plt.title('Интенсивность отказов')
plt.xlabel('Время')
plt.ylabel('Интенсивность')
plt.legend()
plt.grid()

# Плотность распределения времени до отказа
plt.figure(figsize=(10, 6))
plt.plot(t, pdf_uniform(t, a1, b1), label='Uniform U(100, 5000)')
plt.plot(t, pdf_normal(t, mu2, sigma2), label='Normal N(500, 10000)')
plt.plot(t, pdf_gamma(t, k3, theta3), label='Gamma Γ(8, 65)')
plt.title('Плотность распределения времени до отказа')
plt.xlabel('Время')
plt.ylabel('Плотность')
plt.legend()
plt.grid()

# Гамма-процентная наработка до отказа
gamma_values = np.arange(0, 101, 10)
gamma_percentiles_uniform = [gamma_percentile_uniform(g, a1, b1) for g in gamma_values]
gamma_percentiles_normal = [gamma_percentile_normal(g, mu2, sigma2) for g in gamma_values]
gamma_percentiles_gamma = [gamma_percentile_gamma(g, k3, theta3) for g in gamma_values]

plt.figure(figsize=(10, 6))
plt.plot(gamma_values, gamma_percentiles_uniform, label='Uniform U(100, 5000)')
plt.plot(gamma_values, gamma_percentiles_normal, label='Normal N(500, 10000)')
plt.plot(gamma_values, gamma_percentiles_gamma, label='Gamma Γ(8, 65)')
plt.title('Гамма-процентная наработка до отказа')
plt.xlabel('Гамма (%)')
plt.ylabel('Время')
plt.legend()
plt.grid()


# Вывод результатов
print("Средняя наработка до отказа:")
print(f"Uniform U(100, 5000): {mean_uniform}")
print(f"Normal N(500, 10000): {mean_normal}")
print(f"Gamma Γ(8, 65): {mean_gamma}")

print("\nДисперсия и среднее квадратическое отклонение:")
print(f"Uniform U(100, 5000): Дисперсия = {var_uniform}, СКО = {std_uniform}")
print(f"Normal N(500, 10000): Дисперсия = {var_normal}, СКО = {std_normal}")
print(f"Gamma Γ(8, 65): Дисперсия = {var_gamma}, СКО = {std_gamma}")

plt.show()
