using LinearAlgebra # Матричные операции
using QuadGK # Численное интегрирование
using Printf
using Roots

# Параметры задачи
A = [-32.0 -18.75 -22.5;
      37.0  21.0   29.0;
       7.0   4.25   4.5]

B = [-2.5; 4.0; 0.5]
C = [-3.0; -1.5; -6.0]
K = [1.5; -2.0; -0.5]

m1 = -5.0
m2 = 3.842556
ell2 = 6.629453
Tr = (3 * π) / 2

f(t) = 0.5 + sin(2t + 1.815774) + 3sin(4t + 1.62)

# Целевая функция F(x) из уравнения (3)
const E3 = Matrix{Float64}(I, 3, 3)
const M  = inv(E3 - exp(A * Tr))

function F(x::Float64)
    # I1 и I2 — векторные интегралы в формуле (3)
    I1, _ = quadgk(ξ -> exp(-A * ξ) * (B * m1 + K * f(ξ)), 0.0, x)
    I2, _ = quadgk(ξ -> exp(A * ξ) * (B * m2 + K * f(Tr - ξ)), 0.0, Tr - x)
    return dot(C, M * exp(A * x) * (I1 + I2)) - ell2
end


# Счётчик вызовов F(x)
mutable struct EvalCounter
    n::Int
end

counted(F::Function, c::EvalCounter) = x -> begin
    c.n += 1
    return F(x)
end

# Численные производные
dF(F::Function, x::Float64; h::Float64 = 1e-6) =
    (F(x + h) - F(x - h)) / (2h)

ddF(F::Function, x::Float64; h::Float64 = 1e-5) =
    (F(x + h) - 2F(x) + F(x - h)) / (h^2)

# Структура для хранения результата
mutable struct Result
    name::String
    root::Float64
    iters::Int
    fevals::Int
end


# 1. Bisection
function bisection(F::Function, a::Float64, b::Float64; tol=1e-12, maxiter=100)
    fa, fb = F(a), F(b)
    @assert fa * fb < 0 "Bisection: no sign change on the interval"
    it = 0
    while (b - a) / 2 > tol && it < maxiter
        m = a + (b - a) / 2 # (a + b) / 2
        fm = F(m)
        if fa * fm <= 0
            b, fb = m, fm
        else
            a, fa = m, fm
        end
        it += 1
    end
    return (a + (b - a) / 2), it
end

# 2. Trisection
function trisection(F::Function, a::Float64, b::Float64; tol=1e-12, maxiter=100)
    fa, fb = F(a), F(b)
    @assert fa * fb < 0 "Trisection: no sign change on the interval"
    it = 0
    while (b - a) > tol && it < maxiter
        x1 = a + (b - a) / 3
        x2 = a + 2 * (b - a) / 3
        f1, f2 = F(x1), F(x2)

        if fa * f1 <= 0
            b, fb = x1, f1
        elseif f1 * f2 <= 0
            a, fa = x1, f1
            b, fb = x2, f2
        else
            a, fa = x2, f2
        end
        it += 1
    end
    return (a + (b - a) / 2), it
end

# 3. False Position
function false_position(F::Function, a::Float64, b::Float64; tol=1e-12, maxiter=100)
    fa, fb = F(a), F(b)
    @assert fa * fb < 0 "False position: no sign change on the interval"
    it = 0
    c = a
    while it < maxiter
        c = b - fb * (b - a) / (fb - fa)
        fc = F(c)
        if abs(fc) < tol
            return c, it + 1
        end
        if fa * fc < 0
            b, fb = c, fc
        else
            a, fa = c, fc
        end
        it += 1
    end
    return c, it
end

# 4. Newton
function newton(F::Function, x0::Float64; tol=1e-12, maxiter=100)
    x = x0
    it = 0
    while it < maxiter
        fx = F(x)
        abs(fx) < tol && return x, it
        dfx = dF(F, x)
        xnew = x - fx / dfx
        if abs(xnew - x) < tol
            return xnew, it + 1
        end
        x = xnew
        it += 1
    end
    return x, it
end

# 5. Secant
function secant(F::Function, x0::Float64, x1::Float64; tol=1e-12, maxiter=100)
    f0, f1 = F(x0), F(x1)
    it = 0
    while it < maxiter
        denom = f1 - f0
        @assert denom != 0.0 "Secant: zero denominator"
        x2 = x1 - f1 * (x1 - x0) / denom
        if abs(x2 - x1) < tol || abs(F(x2)) < tol
            return x2, it + 1
        end
        x0, f0 = x1, f1
        x1, f1 = x2, F(x2)
        it += 1
    end
    return x1, it
end

# 6. Modified Secant
function modified_secant(F::Function, x0::Float64; tol=1e-12, maxiter=100)
    x = x0
    it = 0
    while it < maxiter
        fx = F(x)
        abs(fx) < tol && return x, it
        d = fx
        denom = F(x + d) - fx
        @assert denom != 0.0 "Modified Secant: zero denominator"
        xnew = x - d * fx / denom
        if abs(xnew - x) < tol
            return xnew, it + 1
        end
        x = xnew
        it += 1
    end
    return x, it
end

# 7. FP-MSe 
function fp_mse(F::Function, a::Float64, b::Float64; drel=1e-6, tol=1e-12, maxiter=100)
    fa, fb = F(a), F(b)
    @assert fa * fb < 0 "FP-MSe: no sign change on the interval"
    it = 0
    while it < maxiter
        it += 1

        # Шаг false position
        x_fp = a - fa * (b - a) / (fb - fa)
        fx_fp = F(x_fp)

        # Критерий остановки по невязке
        if abs(fx_fp) < tol
            return x_fp, it
        else        
            # Шаг modified secant
            d = fx_fp
            denom = F(x_fp + d) - fx_fp
            @assert denom != 0.0 "FP-MSe: zero denominator"
            x_mse = x_fp - d * fx_fp / denom
            fx_mse = F(x_mse)
            
            # Выбор лучшего нового приближения
            if a < x_mse < b && abs(fx_mse) < abs(fx_fp)
                if fa * fx_mse < 0
                    b, fb = x_mse, fx_mse
                else 
                    a, fa = x_mse, fx_mse
                end
            else
                if fa * fx_fp < 0
                    b, fb = x_fp, fx_fp
                else
                    a, fa = x_fp, fx_fp
                end
            end
        end
    end
    
    return (a + b) / 2, maxiter
end

# 8. Halley
function halley(F::Function, x0::Float64; tol=1e-12, maxiter=100)
    x = x0
    it = 0
    while it < maxiter
        fx = F(x)
        abs(fx) < tol && return x, it
        dfx = dF(F, x)
        ddfx = ddF(F, x)
        denom = 2 * dfx^2 - fx * ddfx
        @assert denom != 0.0 "Halley: zero denominator"
        xnew = x - (2 * fx * dfx) / denom
        if abs(xnew - x) < tol
            return xnew, it + 1
        end
        x = xnew
        it += 1
    end
    return x, it
end

# 9. Ridders
function ridders(F::Function, a::Float64, b::Float64; tol=1e-12, maxiter=100)
    fa, fb = F(a), F(b)
    @assert fa * fb < 0 "Ridders: no sign change on the interval"
    it = 0
    while it < maxiter
        m = a + (b - a) / 2 # (a + b) / 2
        fm = F(m)
        s2 = fm^2 - fa * fb
        if s2 <= 0
            return m, it + 1
        end
        s = sqrt(s2)
        x = m + (m - a) * sign(fa - fb) * fm / s
        fx = F(x)

        if abs(fx) < tol || abs(b - a) < tol
            return x, it + 1
        end

        if fm * fx < 0
            a, fa = m, fm
            b, fb = x, fx
        elseif fa * fx < 0
            b, fb = x, fx
        else
            a, fa = x, fx
        end

        it += 1
    end
    return (a + (b - a) / 2), it
end

function brent(F::Function, a::Float64, b::Float64; tol=1e-12, maxiter=100)
    fa, fb = F(a), F(b)
    @assert fa * fb < 0 "Brent: no sign change on the interval"

    if abs(fa) < abs(fb)
        a, b = b, a
        fa, fb = fb, fa
    end

    c, fc = a, fa
    d_prev = a # Переменная для отслеживания предыдущего шага
    mflag = true # На предыдущей итерации алгоритм использовал метод дихотомии
    s = b

    for it in 1:maxiter
        # Вычисляем параметры R, S, T
        if fa != fc && fb != fc
            # Обратная квадратичная интерполяция
            R = fb / fc
            S = fb / fa
            T = fa / fc
            
            P = S * (T * (R - T) * (c - b) - (1.0 - R) * (b - a))
            Q = (T - 1.0) * (R - 1.0) * (S - 1.0)
        else
            # Метод секущих
            S = fb / fa
            P = S * (c - b)
            Q = S - 1.0
        end

        # Потенциальное новое приближение (x = b + P/Q)
        d = P / Q
        s = b + d

        # Условия переключения на метод дихотомии
        low = min(a, b)
        high = max(a, b)

        cond1 = !(low <= s <= high)                     # Выход за границы
        cond2 = mflag && abs(d) >= abs(b - c) / 2       # Шаг интерполяции не сузил отрезок в 2 раза (после дихотомии)
        cond3 = !mflag && abs(d) >= abs(c - d_prev) / 2 # Шаг интерполяции не сузил отрезок в 2 раза (после интерполяции)
        cond4 = mflag && abs(b - c) < tol               # Текущий отрезок уже слишком мал (на уровне погрешности)
        cond5 = !mflag && abs(c - d_prev) < tol         # Предыдущий шаг был слишком мал (защита от бесконечного цикла)

        if cond1 || cond2 || cond3 || cond4 || cond5
            # Метод дихотомии
            s = (a + (b - a) / 2)
            mflag = true
        else
            mflag = false
        end

        fs = F(s)
        d_prev, c = c, b
        fc = fb

        # Обновление интервала локализации
        if fa * fs < 0
            b, fb = s, fs
        else
            a, fa = s, fs
        end

        # Точка b всегда должна быть лучшим приближением
        if abs(fa) < abs(fb)
            a, b = b, a
            fa, fb = fb, fa
        end

        # Проверка критериев останова
        if abs(fb) < tol || abs(b - a) < tol
            return b, it
        end
    end

    return b, maxiter
end

# 11. Steffensen
function steffensen(F::Function, x0::Float64; tol=1e-12, maxiter=100)
    x = x0
    it = 0
    while it < maxiter
        fx = F(x)
        abs(fx) < tol && return x, it
        denom = F(x + fx) - fx
        @assert denom != 0.0 "Steffensen: zero denominator"
        xnew = x - fx^2 / denom
        if abs(xnew - x) < tol
            return xnew, it + 1
        end
        x = xnew
        it += 1
    end
    return x, it
end

# 12. Modified Steffensen
function modified_steffensen(F::Function, x0::Float64; tol=1e-12, maxiter=100, gamma0=1.0)
    x = x0
    gamma = gamma0                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    it = 0
    
    # Переменные для хранения памяти с предыдущей итерации
    x_prev = 0.0
    w_prev = 0.0
    fx_prev = 0.0
    fw_prev = 0.0

    while it < maxiter
        fx = F(x)
        abs(fx) < tol && return x, it

        # Начиная со второй итерации (k >= 1), вычисляем новый gamma_k
        if it > 0
            # Разделенные разности 1-го порядка
            f_x_xprev = (fx - fx_prev) / (x - x_prev)
            f_xprev_wprev = (fx_prev - fw_prev) / (x_prev - w_prev)
            
            # Разделенная разность 2-го порядка
            f_x_xprev_wprev = (f_x_xprev - f_xprev_wprev) / (x - w_prev)
            
            # Аппроксимация производной: N2'(x_k)
            n2_prime = f_x_xprev + f_x_xprev_wprev * (x - x_prev)
            
            @assert n2_prime != 0.0 "Modified Steffensen: zero derivative approximation (N2'(x_k) = 0)"
            gamma = -1.0 / n2_prime
        end

        # Вычисление узла w_k и значения функции в нем
        w = x + gamma * fx
        fw = F(w)

        denom = fw - fx
        @assert denom != 0.0 "Modified Steffensen: zero denominator"

        # Итерационный шаг
        xnew = x - gamma * fx^2 / denom

        if abs(xnew - x) < tol
            return xnew, it + 1
        end

        # Сохранение текущих узлов в память для следующей итерации
        x_prev = x
        w_prev = w
        fx_prev = fx
        fw_prev = fw

        x = xnew
        it += 1
    end

    return x, it
end

# Запуск всех методов и вывод таблицы
function run_all()
    tol = 1e-12
    a, b = 0.0, Tr

    x0 = 1.5
    x1 = 2.0

    results = Result[]

    # Встроенные в Roots.jl методы используются как проверка
    c = EvalCounter(0); f = counted(F, c); root, it = bisection(f, a, b; tol=tol); push!(results, Result("Bi", root, it, c.n))
    c = EvalCounter(0); f = counted(F, c); root, it = trisection(f, a, b; tol=tol); push!(results, Result("Tri", root, it, c.n))
    c = EvalCounter(0); f = counted(F, c); root, it = false_position(f, a, b; tol=tol); push!(results, Result("FP", root, it, c.n))
    c = EvalCounter(0); f = counted(F, c); root, it = newton(f, x0; tol=tol); push!(results, Result("NR", root, it, c.n))
    c = EvalCounter(0); f = counted(F, c); root, it = secant(f, x0, x1; tol=tol); push!(results, Result("Se", root, it, c.n))
    c = EvalCounter(0); f = counted(F, c); root, it = modified_secant(f, x0; tol=tol); push!(results, Result("MSe", root, it, c.n))
    c = EvalCounter(0); f = counted(F, c); root, it = fp_mse(f, a, b; tol=tol); push!(results, Result("FP-MSe", root, it, c.n))
    c = EvalCounter(0); f = counted(F, c); root, it = halley(f, x0; tol=tol); push!(results, Result("Halley", root, it, c.n))
    c = EvalCounter(0); f = counted(F, c); root, it = ridders(f, a, b; tol=tol); push!(results, Result("Ridders", root, it, c.n))
    c = EvalCounter(0); f = counted(F, c); root, it = brent(f, a, b; tol=tol); push!(results, Result("VW-Brent", root, it, c.n))
    c = EvalCounter(0); f = counted(F, c); root, it = steffensen(f, x0; tol=tol); push!(results, Result("Steffensen", root, it, c.n))
    c = EvalCounter(0); f = counted(F, c); root, it = modified_steffensen(f, x0; tol=tol); push!(results, Result("MSt", root, it, c.n))

    # Вывод результатов
    println("\nManual implementation results:")
    @printf("%-14s %-20s %-12s %-12s %-16s\n", "Method", "tau1", "iters", "f-evals", "|tau1-pi/2|")
    for r in results
        @printf("%-14s %-20.15f %-12d %-12d %-16.8e\n",
                r.name, r.root, r.iters, r.fevals, abs(r.root - π/2))
    end

    println("\nRoots.jl validation (find_zero):")
    @printf("%-14s %-20s\n", "Method", "tau1")
    @printf("%-14s %-20.15f\n", "Bi", find_zero(F, (a, b), Bisection()))
    @printf("%-14s %-20.15f\n", "FP", find_zero(F, (a, b), FalsePosition()))
    @printf("%-14s %-20.15f\n", "Se", find_zero(F, (x0, x1), Secant()))
    @printf("%-14s %-20.15f\n", "NR", find_zero((F, x -> dF(F, x)), x0, Roots.Newton()))
    @printf("%-14s %-20.15f\n", "Halley", find_zero((F, x -> dF(F, x), x -> ddF(F, x)), x0, Roots.Halley()))
    @printf("%-14s %-20.15f\n", "VW-Brent", find_zero(F, (a, b), Roots.Brent()))
    @printf("%-14s %-20.15f\n", "Ridders", find_zero(F, (a, b), Roots.Ridders()))
    @printf("%-14s %-20.15f\n", "Steffensen", find_zero(F, x0, Order2()))
end

run_all()
