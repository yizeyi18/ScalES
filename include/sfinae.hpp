#pragma once
#include <type_traits>

namespace dgdft {

template< typename T, typename = std::void_t<> >
struct has_data_member : public std::false_type { };

template <typename T>
struct has_data_member< T, 
  std::void_t<decltype( std::declval<T>().data() )>
> : public std::true_type{ };

template< typename T, typename = std::void_t<> >
struct has_size_member : public std::false_type { };

template <typename T>
struct has_size_member< T, 
  std::void_t<decltype( std::declval<T>().size() )>
> : public std::true_type{ };

template< typename T, typename = std::void_t<> >
struct has_Data_member : public std::false_type { };

template <typename T>
struct has_Data_member< T, 
  std::void_t<decltype( std::declval<T>().Data() )>
> : public std::true_type{ };

template< typename T, typename = std::void_t<> >
struct has_Size_member : public std::false_type { };

template <typename T>
struct has_Size_member< T, 
  std::void_t<decltype( std::declval<T>().Size() )>
> : public std::true_type{ };


template <typename T>
inline constexpr bool has_data_member_v = has_data_member<T>::value;
template <typename T>
inline constexpr bool has_size_member_v = has_size_member<T>::value;
template <typename T>
inline constexpr bool has_Data_member_v = has_Data_member<T>::value;
template <typename T>
inline constexpr bool has_Size_member_v = has_Size_member<T>::value;



template <typename T>
std::enable_if_t< has_data_member_v<T>, decltype(std::declval<T>().data()) > 
  get_data_member( T& container ){ return container.data(); }
template <typename T>
std::enable_if_t< has_data_member_v<T>, decltype(std::declval<T>().Data()) > 
  get_data_member( T& container ){ return container.Data(); }

template <typename T>
std::enable_if_t< has_data_member_v<T>, size_t > 
  get_size_member( T& container ){ return container.size(); }
template <typename T>
std::enable_if_t< has_Size_member_v<T>, size_t > 
  get_size_member( T& container ){ return container.Size(); }



}
