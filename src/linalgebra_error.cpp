export module linalgebra:error;
import std;

export namespace linalgebra {

class LinAlgError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class DimensionMismatchError : public LinAlgError {
public:
    explicit DimensionMismatchError(const std::string& message)
        : LinAlgError(message) {}
};

class SingularMatrixError : public LinAlgError {
public:
    explicit SingularMatrixError(const std::string& message)
        : LinAlgError(message) {}
};

class NonConvergenceError : public LinAlgError {
public:
    explicit NonConvergenceError(const std::string& message)
        : LinAlgError(message) {}
};

}  // namespace linalgebra
