#pragma once

#include <utility>
#include <tuple>
#include <iostream>
#include "tuple.hpp"

template <typename... Types>
decltype(auto) operator>>(std::istream& is, std::tuple<Types...>& tuple)
{
	forEach(tuple, [&is](auto& element) { is >> element; });

	return is;
}

template <typename... Types>
decltype(auto) operator<<(std::ostream& os, const std::tuple<Types...>& tuple)
{
	os << "( ";

	forEach(tuple, [&os](const auto& element) { os << element << " "; });

	os << ")";

	return os;
}

template <typename Type, size_t Size>
decltype(auto) operator>>(std::istream& is, std::array<Type, Size>& array)
{
	forEach(array, [&is](auto& element) { is >> element; });

	return is;
}

template <typename Type, size_t Size>
decltype(auto) operator<<(std::ostream& os, const std::array<Type, Size>& array)
{
	os << "[ ";

	forEach(array, [&os](const auto& element) { os << element << " "; });

	os << "]";

	return os;
}

template <typename First, typename Second>
decltype(auto) operator>>(std::istream& is, std::pair<First, Second>& pair)
{
	is >> pair.first >> pair.second;

	return is;
}

template <typename First, typename Second>
decltype(auto) operator<<(std::ostream& os, const std::pair<First, Second>& pair)
{
	os << "( " << pair.first << " " << pair.second << " )";

	return os;
}

template <typename Container>
auto operator>>(std::istream& is, Container& container) -> decltype(container.begin(), container.end(), is)
{
	for (auto& element : container)
	{
		is >> element;
	}

	return is;
}

template <typename Container>
auto operator<<(std::ostream& os, const Container& container) -> decltype(container.begin(), container.end(), os)
{
	os << "[ ";

	for (const auto& element : container)
	{
		os << element << " ";
	}

	os << "]";

	return os;
}