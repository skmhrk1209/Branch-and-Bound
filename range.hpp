#pragma once

#include <array>
#include <utility>

template <typename Type, Type... numbers>
constexpr auto rangeImpl(std::integer_sequence<Type, numbers...>) -> std::array<Type, sizeof...(numbers)>
{
    return {numbers...};
}
template <auto Size>
constexpr auto range()
{
    return rangeImpl(std::make_integer_sequence<decltype(Size), Size>());
}
