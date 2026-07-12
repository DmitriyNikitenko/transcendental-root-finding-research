using LinearAlgebra
using QuadGK
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

        # Критерий остановки по невязке
        if abs(F(x)) < tol
            return x, it
        else        
            # Шаг modified secant
            fx_fp = F(x_fp)
            d = fx_fp
            denom = F(x_fp + d) - fx_fp
            @assert denom != 0.0 "FP-MSe: zero denominator"
            x_mse = x_fp - d * fx_fp / denom
            
            # Выбор лучшего нового приближения
            if a < x_mse < b && abs(F(x_mse)) < abs(F(x_fp))
                if F(a) * F(x_mse) < 0
                    b = x_mse
                else 
                    a = x_mse
                end
            else
                if F(a) * F(x_fp) < 0
                    b = x_fp
                else
                    a = x_fp
                end
            end
        end
    end
    return x, it
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
