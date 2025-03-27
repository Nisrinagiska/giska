library(MASS)
# Fungsi sigmoid
sigmoid <- function(z) {
  return(1 / (1 + exp(-z)))
}

# Fungsi Regresi Logistik dengan Newton-Raphson
logistic_regression_NR <- function(X, y, tol = 1e-6, max_iter = 100) {
  X <- cbind(1, X)
  beta <- rep(0, ncol(X))

  for (i in 1:max_iter) {
    p <- sigmoid(X %*% beta)
    epsilon <- 1e-6  # Mencegah singularitas
    W <- diag(as.vector(p * (1 - p) + epsilon))

    # Hitung gradient dan Hessian
    gradient <- t(X) %*% (y - p)
    Hessian <- -t(X) %*% W %*% X

    # Update beta
    beta_new <- beta - solve(Hessian) %*% gradient

    # Cek konvergensi
    if (max(abs(beta_new - beta)) < tol) break

    beta <- beta_new
  }

  return(list(method = "Newton-Raphson", beta = beta, fit = sigmoid(X %*% beta)))
}

# Fungsi Regresi Logistik dengan IRLS
logistic_regression_IRLS <- function(X, y, tol = 1e-6, max_iter = 100) {
  X <- cbind(1, X)  # Tambahkan intercept
  beta <- rep(0, ncol(X))  # Inisialisasi beta

  for (i in 1:max_iter) {
    p <- sigmoid(X %*% beta)
    epsilon <- 1e-6  # Mencegah singularitas
    W <- diag(as.vector(p * (1 - p) + epsilon))

    # Update beta dengan metode IRLS
    z <- X %*% beta + (y - p)
    beta_new <- solve(t(X) %*% W %*% X) %*% t(X) %*% W %*% z

    # Cek konvergensi
    if (max(abs(beta_new - beta)) < tol) break

    beta <- beta_new
  }

  return(list(method = "IRLS", beta = beta, fit = sigmoid(X %*% beta)))
}

# Contoh dataset
set.seed(123)
n <- 100
X <- matrix(rnorm(n * 2), ncol = 2)
y <- rbinom(n, 1, prob = 0.5)

# Regresi logistik dengan NR & IRLS
result_NR <- logistic_regression_NR(X, y)
result_IRLS <- logistic_regression_IRLS(X, y)

# Tampilkan hasil
print(result_NR)
print(result_IRLS)

