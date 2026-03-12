#include "linalg_error.hpp"
#include "vector.hpp"

#include <catch2/catch_test_macros.hpp>

#include <type_traits>
#include <utility>

using linalg::DimensionMismatchError;
using linalg::Vector;

TEST_CASE("Vector constructors initialize size and values", "[vector]") {
    const Vector empty;
    REQUIRE(empty.size() == 0);
    REQUIRE(empty.empty());

    const Vector zeroed(4);
    REQUIRE(zeroed.size() == 4);
    for (std::size_t i = 0; i < zeroed.size(); ++i) {
        CHECK(zeroed[i] == 0.0);
    }

    const Vector filled(3, 2.5);
    REQUIRE(filled.size() == 3);
    CHECK(filled[0] == 2.5);
    CHECK(filled[1] == 2.5);
    CHECK(filled[2] == 2.5);
}

TEST_CASE("Vector supports checked element access and fill", "[vector]") {
    Vector v{1.0, 2.0, 3.0};
    REQUIRE(v.size() == 3);

    v[1] = 7.0;
    CHECK(v[0] == 1.0);
    CHECK(v[1] == 7.0);
    CHECK(v[2] == 3.0);

    v.fill(-2.0);
    CHECK(v[0] == -2.0);
    CHECK(v[1] == -2.0);
    CHECK(v[2] == -2.0);

    CHECK_THROWS_AS(v[3], std::out_of_range);
}

TEST_CASE("Vector copy and move preserve values", "[vector]") {
    static_assert(std::is_nothrow_move_constructible_v<Vector>);

    Vector original{4.0, -1.0, 8.0};
    Vector copied = original;
    copied[0] = 10.0;

    CHECK(original[0] == 4.0);
    CHECK(copied[0] == 10.0);

    Vector moved = std::move(original);
    REQUIRE(moved.size() == 3);
    CHECK(moved[0] == 4.0);
    CHECK(moved[1] == -1.0);
    CHECK(moved[2] == 8.0);
}

TEST_CASE("Vector arithmetic enforces shape compatibility", "[vector]") {
    const Vector a{1.0, 2.0, 3.0};
    const Vector b{4.0, 5.0, 6.0};

    const Vector sum = a + b;
    CHECK(sum[0] == 5.0);
    CHECK(sum[1] == 7.0);
    CHECK(sum[2] == 9.0);

    const Vector diff = b - a;
    CHECK(diff[0] == 3.0);
    CHECK(diff[1] == 3.0);
    CHECK(diff[2] == 3.0);

    const Vector scaled = 0.5 * a;
    CHECK(scaled[0] == 0.5);
    CHECK(scaled[1] == 1.0);
    CHECK(scaled[2] == 1.5);

    CHECK(linalg::dot(a, b) == 32.0);

    const Vector short_vec{1.0, 2.0};
    CHECK_THROWS_AS(a + short_vec, DimensionMismatchError);
    CHECK_THROWS_AS(linalg::dot(a, short_vec), DimensionMismatchError);
}
