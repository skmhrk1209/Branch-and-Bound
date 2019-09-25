#pragma once

#include <functional>
#include <type_traits>
#include <utility>

// currying & partial application in C++17
template <typename Function>
constexpr auto curry(Function &&function)
{
    return [=](auto &&... args1) {
        if constexpr (std::is_invocable<decltype(function), decltype(args1)...>::value)
        {
            return function(std::forward<decltype(args1)>(args1)...);
        }
        else
        {
            return curry([=](auto &&... args2) -> decltype(function(args1..., std::forward<decltype(args2)>(args2)...)) {
                return function(args1..., std::forward<decltype(args2)>(args2)...);
            });
        }
    };
}

// fixed point combinator for anonymous recursive function
template <typename Function>
constexpr auto fix(Function &&function)
{
    return curry(function)(function);
}