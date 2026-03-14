#pragma once

#include <stdexcept>
#include <string>

namespace linalg {

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

}  // namespace linalg
