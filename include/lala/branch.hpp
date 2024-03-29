// Copyright 2022 Pierre Talbot

#ifndef LALA_POWER_BRANCH_HPP
#define LALA_POWER_BRANCH_HPP

#include "battery/vector.hpp"
#include "lala/logic/logic.hpp"

namespace lala {

template <class TellTy, class Alloc>
class Branch {
public:
  using tell_type = TellTy;
  using allocator_type = Alloc;

  template <class TellTy2, class Alloc2>
  friend class Branch;

private:
  battery::vector<tell_type, allocator_type> children;
  int current_idx;

public:
  CUDA Branch(const allocator_type& alloc = allocator_type()): children(alloc), current_idx(-1) {}
  Branch(const Branch&) = default;
  Branch(Branch&&) = default;
  CUDA Branch(battery::vector<tell_type, allocator_type>&& children)
   : children(std::move(children)), current_idx(-1) {}

  template <class BranchType>
  CUDA Branch(const BranchType& branch, const allocator_type& alloc = allocator_type())
   : children(branch.children, alloc), current_idx(branch.current_idx) {}

  CUDA int size() const {
    return static_cast<int>(children.size()/*size_t*/);
  }

  CUDA const tell_type& next() {
    assert(has_next());
    return children[++current_idx];
  }

  CUDA const tell_type& operator[](size_t idx) {
    return children[idx];
  }

  CUDA bool has_next() const {
    return current_idx + 1 < size();
  }

  CUDA void prune() {
    current_idx = size();
  }

  CUDA bool is_pruned() const {
    return current_idx >= size();
  }

  CUDA const tell_type& current() const {
    assert(current_idx != -1 && current_idx < children.size());
    return children[current_idx];
  }
};

}

#endif
